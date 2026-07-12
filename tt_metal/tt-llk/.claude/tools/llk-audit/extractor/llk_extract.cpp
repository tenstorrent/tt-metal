// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// llk_extract — the C++ front end of the LLK race-audit recall tool.
//
// It does ONE job: parse an LLK header with a real Clang AST + preprocessor and
// emit a *generic, semantics-free* JSON "fact base" of everything a race-audit
// checker might need. All hazard-specific meaning (which function produces a
// cfg pointer, which macro is a "consumer", which call is a semaphore wait)
// lives in the Python `registry.py`, NOT here. That separation is deliberate:
//
//   * The fragile, fidelity-critical work (parsing template/macro-heavy headers,
//     tracing pointer provenance through the AST, recovering macro names+args
//     the AST discards) needs libTooling and only needs to be gotten right once.
//   * The frequently-edited work (what names/signatures count as what) is data,
//     kept in the Python registry so that when an LLK signature changes you edit
//     one table, never this file.
//
// Facts emitted (all filtered to the configured --path-filter, default tt_llk_):
//   functions       every function definition + its source range
//   pointer_writes  assignments through a subscript/deref lvalue, with the
//                   PROVENANCE of the base pointer (the name of the function or
//                   variable that produced it) — the registry maps that name to
//                   a write kind (cfg32/cfg16/regfile_gpr/...)
//   calls           every call: callee name, callee source text (so template
//                   args like cfg_reg_rmw_tensix<FIELD> survive), first arg text
//   macros          macro expansions: every FUNCTION-LIKE macro, PLUS the
//                   object-like instruction macros (TTI_NOP/TTI_SFPNOP/...) and
//                   *_RMW field aliases (see MacroPass) — name + full text (args
//                   included). This is how the TTI_*/SEM*/... primitives, which
//                   the AST erases, are recovered
//
// Each fact carries its enclosing function name and a file:line:offset. The
// offset lets the Python side reason about ordering ("does a guard follow the
// write within this function?") deterministically.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/MacroArgs.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace clang::tooling;

static llvm::cl::OptionCategory ToolCat("llk-extract options");
static llvm::cl::opt<std::string> PathFilter(
    "path-filter",
    llvm::cl::desc("Only emit facts whose file path contains this substring "
                   "(default: tt_llk_)"),
    llvm::cl::init("tt_llk_"),
    llvm::cl::cat(ToolCat));
static llvm::cl::opt<std::string> ArchTag("arch", llvm::cl::desc("Architecture tag echoed into the JSON"), llvm::cl::init(""), llvm::cl::cat(ToolCat));

namespace
{

// A raw fact. `kind` distinguishes the four fact families; the remaining fields
// are populated as relevant per family (empty otherwise). Keeping one struct
// keeps ordering/attribution uniform.
struct Fact
{
    std::string family; // "function" | "pointer_write" | "call" | "macro"
    std::string file;
    unsigned line   = 0;
    unsigned off    = 0;        // file offset of the (spelling) location
    unsigned endOff = 0;        // functions only: end of body
    std::string function;       // enclosing function (resolved in a post-pass)
    std::string name;           // function name / callee ident / macro name
    std::string text;           // macro full text / callee source text
    std::string op;             // pointer_write: assignment operator ("=", "|=", ...)
    std::string provenanceKind; // pointer_write: "call" | "var" | "cast"
    std::string producer;       // pointer_write: init-callee name / var name / cast text
    std::string indexText;      // pointer_write: subscript index expression
    std::string arg0;           // call: first argument source text
    int argc = -1;              // call: explicit argument count (disambiguates overloads)
    std::string recv;           // call: member-call receiver expr text (e.g. "cb_buf")
    std::string recvType;       // call: member-call receiver TYPE name (e.g. "CircularBuffer")
};

struct State
{
    SourceManager *SM     = nullptr;
    const LangOptions *LO = nullptr;
    std::vector<Fact> facts;

    // Function ranges collected first, used to attribute every other fact to its
    // enclosing function.
    struct FnRange
    {
        std::string name, file;
        unsigned b, e;
    };

    std::vector<FnRange> fns;
    unsigned parseErrors = 0;
};

bool inScope(const State &S, SourceLocation L, std::string &file, unsigned &line, unsigned &off)
{
    SourceLocation Sp = S.SM->getSpellingLoc(L);
    if (Sp.isInvalid())
    {
        return false;
    }
    llvm::StringRef f = S.SM->getFilename(Sp);
    if (!f.contains(PathFilter) || f.contains("/tests/"))
    {
        return false;
    }
    file = f.str();
    line = S.SM->getSpellingLineNumber(Sp);
    off  = S.SM->getFileOffset(Sp);
    return true;
}

std::string srcText(const State &S, SourceRange R)
{
    return Lexer::getSourceText(CharSourceRange::getTokenRange(R), *S.SM, *S.LO).str();
}

// ---- Preprocessor pass: recover macro invocations the AST loses -------------
// TTI_*/TT_*/SEM*/... are function-like macros; their names+args are gone by the
// time the AST exists. We record every FUNCTION-LIKE macro expansion. Most
// object-like macros are bare constants (register-address names) that already
// show up as pointer_write index text, so they are skipped as noise — EXCEPT the
// `*_RMW` composite aliases used by `cfg_rmw(FIELD_RMW, ...)`: `FIELD_RMW`
// expands to `FIELD_ADDR32, FIELD_SHAMT, FIELD_MASK` BEFORE the AST, so the field
// name is otherwise lost (the AST sees only the expanded numeric args). Capturing
// the `*_RMW` expansion by name lets the Python side resolve the target word.
class MacroPass : public PPCallbacks
{
    State &S;

public:
    explicit MacroPass(State &s) : S(s)
    {
    }

    void MacroExpands(const Token &NameTok, const MacroDefinition &MD, SourceRange Range, const MacroArgs *) override
    {
        const MacroInfo *MI = MD.getMacroInfo();
        if (!MI)
        {
            return;
        }
        const IdentifierInfo *II = NameTok.getIdentifierInfo();
        if (!II)
        {
            return;
        }
        llvm::StringRef nm = II->getName();
        // Denylist encoding-constant / internal expansion macros: TT_OP_* are the
        // opcode-VALUE constants (not an issued instruction) and INSTRUCTION_WORD
        // is expanded INSIDE the real instruction macros — both otherwise get
        // recorded mislocated at their #define site and add noise.
        if (nm.starts_with("TT_OP_") || nm == "INSTRUCTION_WORD")
        {
            return;
        }
        // Capture function-like macros AND object-like INSTRUCTION macros
        // (TTI_NOP, TTI_TRNSPSRCA/B, TTI_SFPNOP, … carry no args but ARE real
        // Tensix instructions the AST erases), plus the *_RMW composite aliases.
        const bool instrLike = nm.starts_with("TTI_") || nm.starts_with("TT_");
        const bool wanted    = MI->isFunctionLike() || instrLike || nm.ends_with("_RMW");
        if (!wanted)
        {
            return;
        }
        Fact f;
        if (!inScope(S, Range.getBegin(), f.file, f.line, f.off))
        {
            return;
        }
        f.family = "macro";
        f.name   = II->getName().str();
        f.text   = srcText(S, Range);
        S.facts.push_back(std::move(f));
    }
};

// ---- AST pass: functions, pointer writes, calls -----------------------------
class Visitor : public RecursiveASTVisitor<Visitor>
{
    State &S;

    // Trace a base pointer expression to the thing that produced it, so the
    // registry can decide what register space it points at. We report the raw
    // producer name; we do NOT decide its meaning here.
    //   - initialized from a call    -> provenanceKind="call", producer=callee
    //   - a plain variable           -> provenanceKind="var",  producer=var name
    //   - a reinterpret/C-style cast -> provenanceKind="cast", producer=cast text
    void provenance(const Expr *Base, std::string &kind, std::string &producer)
    {
        const Expr *E = Base->IgnoreParenImpCasts();
        if (const auto *CE = dyn_cast<CallExpr>(E))
        {
            kind = "call";
            if (const FunctionDecl *FD = CE->getDirectCallee())
            {
                producer = FD->getNameAsString();
            }
            return;
        }
        if (isa<CXXReinterpretCastExpr>(E) || isa<CStyleCastExpr>(E))
        {
            kind     = "cast";
            producer = srcText(S, E->getSourceRange());
            return;
        }
        if (const auto *DRE = dyn_cast<DeclRefExpr>(E))
        {
            if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl()))
            {
                // Prefer the variable's initializer provenance (e.g. cfg = get_cfg_pointer()).
                if (const Expr *Init = VD->getInit())
                {
                    const Expr *IE = Init->IgnoreParenImpCasts();
                    if (const auto *ICE = dyn_cast<CallExpr>(IE))
                    {
                        kind = "call";
                        if (const FunctionDecl *FD = ICE->getDirectCallee())
                        {
                            producer = FD->getNameAsString();
                        }
                        return;
                    }
                    if (isa<CXXReinterpretCastExpr>(IE) || isa<CStyleCastExpr>(IE))
                    {
                        kind     = "cast";
                        producer = srcText(S, IE->getSourceRange());
                        return;
                    }
                }
                kind     = "var";
                producer = VD->getNameAsString();
            }
        }
    }

public:
    explicit Visitor(State &s) : S(s)
    {
    }

    bool VisitFunctionDecl(FunctionDecl *FD)
    {
        if (!FD->doesThisDeclarationHaveABody())
        {
            return true;
        }
        std::string file;
        unsigned line, off;
        if (!inScope(S, FD->getBeginLoc(), file, line, off))
        {
            return true;
        }
        // The end offset must be in the SAME file as the begin offset, else the
        // [b,e] range compares offsets across two files and mis-attributes facts.
        // If the end's spelling is in a different file (macro-wrapped body), fall
        // back to `off` (the Python FactBase then extends it to the next function).
        SourceLocation B = S.SM->getSpellingLoc(FD->getBeginLoc());
        SourceLocation E = S.SM->getSpellingLoc(FD->getEndLoc());
        unsigned endOff  = (E.isValid() && S.SM->getFileID(E) == S.SM->getFileID(B)) ? S.SM->getFileOffset(E) : off;
        State::FnRange fr {FD->getNameAsString(), file, off, endOff};
        S.fns.push_back(fr);
        Fact f;
        f.family = "function";
        f.file   = file;
        f.line   = line;
        f.off    = off;
        f.endOff = fr.e;
        f.name   = fr.name;
        S.facts.push_back(std::move(f));
        return true;
    }

    bool VisitBinaryOperator(BinaryOperator *BO)
    {
        if (!BO->isAssignmentOp())
        {
            return true;
        }
        const Expr *lhs  = BO->getLHS()->IgnoreParenImpCasts();
        const Expr *base = nullptr;
        std::string indexText;
        if (const auto *ASE = dyn_cast<ArraySubscriptExpr>(lhs))
        {
            base      = ASE->getBase();
            indexText = srcText(S, ASE->getIdx()->getSourceRange());
        }
        else if (const auto *UO = dyn_cast<UnaryOperator>(lhs))
        {
            if (UO->getOpcode() == UO_Deref)
            {
                base = UO->getSubExpr();
            }
        }
        if (!base)
        {
            return true;
        }
        Fact f;
        if (!inScope(S, BO->getOperatorLoc(), f.file, f.line, f.off))
        {
            return true;
        }
        f.family    = "pointer_write";
        f.op        = BinaryOperator::getOpcodeStr(BO->getOpcode()).str();
        f.indexText = indexText;
        provenance(base, f.provenanceKind, f.producer);
        if (f.producer.empty())
        {
            // Provenance didn't resolve to a named producer (template-dependent
            // accessor, macro-generated cast, non-VarDecl base). Emit the write
            // anyway, marked "unresolved", so it is VISIBLE rather than silently
            // dropped. Force the kind to "unresolved" so the registry never
            // MIS-classifies it from the fallback text (it will simply ignore it).
            f.provenanceKind = "unresolved";
            f.producer       = srcText(S, base->getSourceRange());
            if (f.producer.empty())
            {
                f.producer = "<unresolved>";
            }
        }
        S.facts.push_back(std::move(f));
        return true;
    }

    bool VisitCallExpr(CallExpr *CE)
    {
        const FunctionDecl *FD = CE->getDirectCallee();
        std::string callee     = FD && FD->getIdentifier() ? FD->getName().str() : "";
        // Keep the callee's written text too, so template args (e.g.
        // cfg_reg_rmw_tensix<FIELD>) survive for the registry to parse.
        Fact f;
        if (!inScope(S, CE->getBeginLoc(), f.file, f.line, f.off))
        {
            return true;
        }
        f.family = "call";
        f.name   = callee;
        f.text   = srcText(S, CE->getCallee()->getSourceRange());
        if (callee.empty() && f.text.empty())
        {
            return true;
        }
        f.argc = static_cast<int>(CE->getNumArgs());
        if (CE->getNumArgs() >= 1)
        {
            f.arg0 = srcText(S, CE->getArg(0)->getSourceRange());
        }
        // Object/method-style API (modern ttnn kernels): cb_buf.reserve_back(1),
        // noc.async_read(...). The callee name is the METHOD ("reserve_back"), so
        // also capture the RECEIVER expr text (for per-object grouping — the CB id
        // is the receiver, not arg0) and the receiver TYPE (to disambiguate a CB
        // method from an unrelated same-named method like std::vector::push_back).
        if (const auto *MCE = dyn_cast<CXXMemberCallExpr>(CE))
        {
            if (const Expr *Obj = MCE->getImplicitObjectArgument())
            {
                f.recv = srcText(S, Obj->getSourceRange());
                // getObjectType() computes getImplicitObjectArgument()->getType(),
                // so it MUST stay inside this null guard: a null implicit object
                // (the "FIXME: member pointers" fallback) would otherwise crash the
                // extractor and drop every fact for the whole TU — a silent capture
                // hole (false-all-clear). getAsCXXRecordDecl() is likewise gated.
                QualType objTy = MCE->getObjectType();
                if (const CXXRecordDecl *RD = objTy->getAsCXXRecordDecl())
                {
                    f.recvType = RD->getNameAsString();
                }
                else if (!objTy.isNull())
                {
                    f.recvType = objTy.getUnqualifiedType().getAsString();
                }
            }
        }
        S.facts.push_back(std::move(f));
        return true;
    }
};

// ---- Consumer: attribute facts to functions, emit JSON ----------------------
class Emit : public ASTConsumer
{
    State &S;

    const State::FnRange *enclosing(const std::string &file, unsigned off)
    {
        const State::FnRange *best = nullptr;
        for (const auto &fr : S.fns)
        {
            if (fr.file == file && fr.b <= off && off <= fr.e)
            {
                if (!best || fr.b > best->b)
                {
                    best = &fr;
                }
            }
        }
        return best;
    }

public:
    explicit Emit(State &s) : S(s)
    {
    }

    void HandleTranslationUnit(ASTContext &Ctx) override
    {
        S.parseErrors = Ctx.getDiagnostics().getClient()->getNumErrors();
        // Attribute non-function facts to their enclosing function.
        for (auto &f : S.facts)
        {
            if (f.family == "function")
            {
                continue;
            }
            if (const auto *fr = enclosing(f.file, f.off))
            {
                f.function = fr->name;
            }
        }
        llvm::json::Array arr;
        for (const auto &f : S.facts)
        {
            llvm::json::Object o;
            o["family"] = f.family;
            o["file"]   = f.file;
            o["line"]   = (std::int64_t)f.line;
            o["off"]    = (std::int64_t)f.off;
            if (!f.function.empty())
            {
                o["function"] = f.function;
            }
            if (!f.name.empty())
            {
                o["name"] = f.name;
            }
            if (!f.text.empty())
            {
                o["text"] = f.text;
            }
            if (f.family == "function")
            {
                o["end_off"] = (std::int64_t)f.endOff;
            }
            if (f.family == "pointer_write")
            {
                o["op"]              = f.op;
                o["provenance_kind"] = f.provenanceKind;
                o["producer"]        = f.producer;
                o["index_text"]      = f.indexText;
            }
            if (f.family == "call" && !f.arg0.empty())
            {
                o["arg0"] = f.arg0;
            }
            if (f.family == "call" && f.argc >= 0)
            {
                o["argc"] = f.argc;
            }
            if (f.family == "call" && !f.recv.empty())
            {
                o["recv"] = f.recv;
            }
            if (f.family == "call" && !f.recvType.empty())
            {
                o["recv_type"] = f.recvType;
            }
            arr.push_back(std::move(o));
        }
        llvm::json::Object root;
        root["arch"]         = ArchTag.getValue();
        root["parse_errors"] = (std::int64_t)S.parseErrors;
        root["facts"]        = std::move(arr);
        llvm::outs() << llvm::formatv("{0}", llvm::json::Value(std::move(root))) << "\n";
    }
};

class Action : public ASTFrontendAction
{
    State St;

public:
    bool BeginSourceFileAction(CompilerInstance &CI) override
    {
        St    = State {};
        St.SM = &CI.getSourceManager();
        St.LO = &CI.getLangOpts();
        CI.getPreprocessor().addPPCallbacks(std::make_unique<MacroPass>(St));
        return true;
    }

    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, llvm::StringRef) override
    {
        St.SM = &CI.getSourceManager();
        St.LO = &CI.getLangOpts();

        class Run : public ASTConsumer
        {
            State &S;

        public:
            explicit Run(State &s) : S(s)
            {
            }

            void HandleTranslationUnit(ASTContext &Ctx) override
            {
                Visitor(S).TraverseDecl(Ctx.getTranslationUnitDecl());
                Emit(S).HandleTranslationUnit(Ctx);
            }
        };

        return std::make_unique<Run>(St);
    }
};

} // namespace

int main(int argc, const char **argv)
{
    auto Expected = CommonOptionsParser::create(argc, argv, ToolCat);
    if (!Expected)
    {
        llvm::errs() << llvm::toString(Expected.takeError());
        return 1;
    }
    ClangTool Tool(Expected->getCompilations(), Expected->getSourcePathList());
    return Tool.run(newFrontendActionFactory<Action>().get());
}
