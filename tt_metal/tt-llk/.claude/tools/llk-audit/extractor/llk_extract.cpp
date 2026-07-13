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
//   pointer_reads   a volatile-pointer READ inside a loop (the hand-rolled busy-
//                   poll of a remotely-written L1 flag) — consumed by the
//                   noc-l1-invalidate check to flag a missing cache invalidate
//   calls           every call: callee name, callee source text (so template
//                   args like cfg_reg_rmw_tensix<FIELD> survive), first arg text,
//                   arg count, and for a member call the receiver expr + its type
//                   (recv/recv_type — the object/method CB/NoC/Semaphore API)
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
// Single source for the default (used in both the help text and cl::init below).
#define DEFAULT_PATH_FILTER "tt_llk_"
static llvm::cl::opt<std::string> PathFilter(
    "path-filter",
    llvm::cl::desc("Only emit facts whose file path contains this substring "
                   "(default: " DEFAULT_PATH_FILTER ")"),
    llvm::cl::init(DEFAULT_PATH_FILTER),
    llvm::cl::cat(ToolCat));
static llvm::cl::opt<std::string> ArchTag("arch", llvm::cl::desc("Architecture tag echoed into the JSON"), llvm::cl::init(""), llvm::cl::cat(ToolCat));

namespace
{

// A raw fact. `family` distinguishes the five fact families; the remaining fields
// are populated as relevant per family (empty otherwise). Keeping one struct
// keeps ordering/attribution uniform.
struct Fact
{
    std::string family; // "function" | "pointer_write" | "call" | "macro" | "pointer_read"
    std::string file;
    unsigned line   = 0;
    unsigned off    = 0;        // file offset of the (spelling) location
    unsigned endOff = 0;        // functions only: end of body
    std::string function;       // enclosing function (resolved in a post-pass)
    std::string name;           // function name / callee ident / macro name
    std::string text;           // macro full text / callee source text
    std::string op;             // pointer_write: assignment operator ("=", "|=", ...)
    std::string provenanceKind; // pointer_write: "call" | "var" | "cast" | "unresolved"
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
// TTI_*/TT_*/SEM*/... macros' names+args are gone by the time the AST exists. We
// record every FUNCTION-LIKE macro expansion AND the object-like INSTRUCTION macros
// (TTI_NOP/TTI_SFPNOP/... — argless but real instructions; see `instrLike` below).
// Other object-like macros are bare constants (register-address names) that already
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

// Best-effort receiver TYPE from a CONSTRUCTOR-style receiver text, e.g.
// "Semaphore<>(id)" / "Semaphore(id)" -> "Semaphore". Returns "" for a bare variable
// ("cb_index") or anything not starting with a Capitalized identifier immediately
// followed by '<' or '(' — deliberately tight so a dependent member call can only
// pick up a receiver type when the receiver is literally a type construction (never
// mislabels a variable), and the checkers' type gates (Semaphore/CircularBuffer/Noc)
// stay the arbiter.
static std::string recvTypeFromText(const std::string &t)
{
    size_t i = 0;
    while (i < t.size() && (t[i] == ' ' || t[i] == '\t'))
    {
        ++i;
    }
    if (i >= t.size() || t[i] < 'A' || t[i] > 'Z')
    {
        return "";
    }
    size_t j = i;
    while (j < t.size() && ((t[j] >= 'A' && t[j] <= 'Z') || (t[j] >= 'a' && t[j] <= 'z') || (t[j] >= '0' && t[j] <= '9') || t[j] == '_'))
    {
        ++j;
    }
    if (j < t.size() && (t[j] == '<' || t[j] == '('))
    {
        return t.substr(i, j - i);
    }
    return "";
}

// ---- AST pass: functions, pointer writes, pointer reads, calls --------------
class Visitor : public RecursiveASTVisitor<Visitor>
{
    State &S;
    // Lexical loop nesting depth. A volatile-pointer READ is only emitted (as a
    // pointer_read fact) when inside a loop — the hand-rolled busy-poll pattern the
    // noc-l1-invalidate check looks for. Gating on loops also keeps the fact count
    // (and any tt-llk noise) far smaller than emitting every rvalue load.
    int loopDepth = 0;

    // Trace a base pointer expression to the thing that produced it, so the
    // registry can decide what register space it points at. We report the raw
    // producer name; we do NOT decide its meaning here.
    //   - initialized from a call    -> provenanceKind="call", producer=callee
    //   - a plain variable           -> provenanceKind="var",  producer=var name
    //   - a reinterpret/C-style cast -> provenanceKind="cast", producer=cast text
    // (When none of these classify, the caller stamps provenanceKind="unresolved"
    //  with the raw base text as producer — a visible, non-dropped write.)
    // Classify a call/cast producer expr. Returns true (setting kind+producer) if
    // E is a CallExpr or a reinterpret/C-style cast; false otherwise. Used on both
    // a base pointer expr and a variable's initializer (same rule, one place).
    bool classifyProducer(const Expr *E, std::string &kind, std::string &producer)
    {
        if (const auto *CE = dyn_cast<CallExpr>(E))
        {
            kind = "call";
            if (const FunctionDecl *FD = CE->getDirectCallee())
            {
                producer = FD->getNameAsString();
            }
            return true;
        }
        if (isa<CXXReinterpretCastExpr>(E) || isa<CStyleCastExpr>(E))
        {
            kind     = "cast";
            producer = srcText(S, E->getSourceRange());
            return true;
        }
        return false;
    }

    void provenance(const Expr *Base, std::string &kind, std::string &producer)
    {
        const Expr *E = Base->IgnoreParenImpCasts();
        if (classifyProducer(E, kind, producer))
        {
            return;
        }
        if (const auto *DRE = dyn_cast<DeclRefExpr>(E))
        {
            if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl()))
            {
                // Prefer the variable's initializer provenance (e.g. cfg = get_cfg_pointer()).
                if (const Expr *Init = VD->getInit())
                {
                    if (classifyProducer(Init->IgnoreParenImpCasts(), kind, producer))
                    {
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
        // Fallback when there is no resolved direct callee: a TEMPLATE / OVERLOADED /
        // dependent call (getDirectCallee()==null) still has its written name in the
        // AST. Without this, the kernel API — which is almost entirely templates /
        // overloads (cb_push_back, cb_reserve_back, noc_async_read, noc_semaphore_inc,
        // tt_memmove<...>) — is dropped to an EMPTY name, and every name-keyed checker
        // silently misses it (a latent kernel-tier false-negative, esp. under any
        // partial parse). Recover the free-function identifier from the callee expr;
        // member-call methods keep flowing through the recv/recvType path below.
        if (callee.empty())
        {
            const Expr *ce = CE->getCallee()->IgnoreParenImpCasts();
            if (const auto *ULE = dyn_cast<UnresolvedLookupExpr>(ce))
            {
                callee = ULE->getName().getAsString();
            }
            else if (const auto *DRE = dyn_cast<DeclRefExpr>(ce))
            {
                callee = DRE->getNameInfo().getName().getAsString();
            }
            else if (const auto *DSDR = dyn_cast<DependentScopeDeclRefExpr>(ce))
            {
                callee = DSDR->getDeclName().getAsString();
            }
            // Dependent/unresolved MEMBER call (obj.method() where obj is template-
            // typed): the modern ttnn object API (cb.push_back(), Semaphore<>(id).up())
            // — without this the method name is dropped and cb-sync / noc-atomic-exit
            // miss it. The receiver is recovered below.
            else if (const auto *DSM = dyn_cast<CXXDependentScopeMemberExpr>(ce))
            {
                callee = DSM->getMember().getAsString();
            }
            else if (const auto *UME = dyn_cast<UnresolvedMemberExpr>(ce))
            {
                callee = UME->getMemberName().getAsString();
            }
        }
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
                // Guard isNull() FIRST: objTy->getAsCXXRecordDecl() dereferences the
                // QualType (operator-> = getTypePtr()), which asserts/segfaults on a
                // null type. A null object type would otherwise crash the TU.
                if (objTy.isNull())
                {
                    // leave recvType empty
                }
                else if (const CXXRecordDecl *RD = objTy->getAsCXXRecordDecl())
                {
                    f.recvType = RD->getNameAsString();
                }
                else
                {
                    f.recvType = objTy.getUnqualifiedType().getAsString();
                }
            }
        }
        // Dependent/unresolved member call (not a CXXMemberCallExpr): recover the
        // receiver text so per-object grouping + the receiver-type heuristic work.
        if (f.recv.empty())
        {
            const Expr *ce2 = CE->getCallee()->IgnoreParenImpCasts();
            if (const auto *DSM = dyn_cast<CXXDependentScopeMemberExpr>(ce2))
            {
                if (!DSM->isImplicitAccess())
                {
                    f.recv = srcText(S, DSM->getBase()->getSourceRange());
                }
            }
            else if (const auto *UME = dyn_cast<UnresolvedMemberExpr>(ce2))
            {
                if (!UME->isImplicitAccess())
                {
                    f.recv = srcText(S, UME->getBase()->getSourceRange());
                }
            }
        }
        // When the receiver TYPE didn't resolve (dependent parse), recover it from a
        // constructor-style receiver text ("Semaphore<>(id)" -> "Semaphore"). Tightly
        // scoped (see recvTypeFromText) so a bare variable never gets a spurious type.
        if (f.recvType.empty() && !f.recv.empty())
        {
            f.recvType = recvTypeFromText(f.recv);
        }
        S.facts.push_back(std::move(f));
        return true;
    }

    // --- loop nesting: bump loopDepth around each loop body so a volatile-pointer
    // read can tell whether it is a busy-poll (inside a loop) vs a one-shot load.
    bool TraverseWhileStmt(WhileStmt *L)
    {
        ++loopDepth;
        bool r = RecursiveASTVisitor::TraverseWhileStmt(L);
        --loopDepth;
        return r;
    }

    bool TraverseForStmt(ForStmt *L)
    {
        ++loopDepth;
        bool r = RecursiveASTVisitor::TraverseForStmt(L);
        --loopDepth;
        return r;
    }

    bool TraverseDoStmt(DoStmt *L)
    {
        ++loopDepth;
        bool r = RecursiveASTVisitor::TraverseDoStmt(L);
        --loopDepth;
        return r;
    }

    bool TraverseCXXForRangeStmt(CXXForRangeStmt *L)
    {
        ++loopDepth;
        bool r = RecursiveASTVisitor::TraverseCXXForRangeStmt(L);
        --loopDepth;
        return r;
    }

    // A VOLATILE-pointer READ inside a loop: an rvalue load (LValueToRValue) through
    // a `*ptr` / `ptr[i]` whose accessed type is volatile-qualified — the shape of a
    // hand-rolled poll of a remotely-written L1 flag. Emitted as a pointer_read fact
    // (with the same provenance the writes carry, so the checker can see get_semaphore
    // etc.). NOT a write: the LHS of an assignment is not an LValueToRValue cast, so
    // this never double-counts a store.
    bool VisitImplicitCastExpr(ImplicitCastExpr *ICE)
    {
        if (ICE->getCastKind() != CK_LValueToRValue || loopDepth == 0)
        {
            return true;
        }
        const Expr *sub  = ICE->getSubExpr()->IgnoreParens();
        const Expr *base = nullptr;
        std::string indexText;
        if (const auto *ASE = dyn_cast<ArraySubscriptExpr>(sub))
        {
            base      = ASE->getBase();
            indexText = srcText(S, ASE->getIdx()->getSourceRange());
        }
        else if (const auto *UO = dyn_cast<UnaryOperator>(sub))
        {
            if (UO->getOpcode() == UO_Deref)
            {
                base = UO->getSubExpr();
            }
        }
        if (!base || !sub->getType().isVolatileQualified())
        {
            return true; // not a volatile pointer deref/subscript read
        }
        Fact f;
        if (!inScope(S, ICE->getExprLoc(), f.file, f.line, f.off))
        {
            return true;
        }
        f.family    = "pointer_read";
        f.op        = "read";
        f.indexText = indexText;
        provenance(base, f.provenanceKind, f.producer);
        if (f.producer.empty())
        {
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
            if (f.family == "pointer_write" || f.family == "pointer_read")
            {
                o["op"]              = f.op;
                o["provenance_kind"] = f.provenanceKind;
                o["producer"]        = f.producer;
                o["index_text"]      = f.indexText;
            }
            const bool isCall = f.family == "call";
            if (isCall && !f.arg0.empty())
            {
                o["arg0"] = f.arg0;
            }
            if (isCall && f.argc >= 0)
            {
                o["argc"] = f.argc;
            }
            if (isCall && !f.recv.empty())
            {
                o["recv"] = f.recv;
            }
            if (isCall && !f.recvType.empty())
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
