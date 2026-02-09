// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "AttributeProtocolCheck.h"

#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/AST/DeclCXX.h>
#include <clang/AST/Expr.h>
#include <clang/AST/ExprCXX.h>
#include <clang/AST/Stmt.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/ASTMatchers/ASTMatchers.h>
#include <clang/Basic/SourceManager.h>
#include <clang/Lex/Lexer.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

using namespace clang;
using namespace clang::ast_matchers;

namespace tt {
namespace tidy {

AttributeProtocolCheck::AttributeProtocolCheck(llvm::StringRef Name, clang::tidy::ClangTidyContext* Context) :
    ClangTidyCheck(Name, Context) {}

void AttributeProtocolCheck::registerMatchers(MatchFinder* Finder) {
    // Match any CXXRecordDecl (struct/class) that either:
    //  1. Is directly named "tensor_args_t" or "operation_attributes_t", OR
    //  2. Already has attribute_names or attribute_values() defined
    //     (catches structs like HaloParams that are *aliased* as operation_attributes_t)
    Finder->addMatcher(
        cxxRecordDecl(
            isDefinition(),
            anyOf(
                hasName("tensor_args_t"),
                hasName("operation_attributes_t"),
                has(varDecl(hasName("attribute_names"))),
                has(cxxMethodDecl(hasName("attribute_values")))))
            .bind("target_struct"),
        this);
}

// Unwrap expression (handles ImplicitCastExpr, MaterializeTemporaryExpr, etc.)
static const Expr* unwrapExpr(const Expr* E) {
    if (!E) {
        return E;
    }
    while (true) {
        if (const auto* ICE = dyn_cast<ImplicitCastExpr>(E)) {
            E = ICE->getSubExpr();
        } else if (const auto* MTE = dyn_cast<MaterializeTemporaryExpr>(E)) {
            E = MTE->getSubExpr();
        } else if (const auto* BTE = dyn_cast<CXXBindTemporaryExpr>(E)) {
            E = BTE->getSubExpr();
        } else {
            break;
        }
    }
    return E;
}

// Find CallExpr in an expression (unwraps as needed)
static const CallExpr* findCallExpr(const Expr* E) {
    E = unwrapExpr(E);
    if (const auto* CE = dyn_cast<CallExpr>(E)) {
        return CE;
    }
    // Try CXXMemberCallExpr, CXXOperatorCallExpr, etc.
    if (const auto* MCE = dyn_cast<CXXMemberCallExpr>(E)) {
        return MCE;
    }
    if (const auto* OCE = dyn_cast<CXXOperatorCallExpr>(E)) {
        return OCE;
    }
    return nullptr;
}

// Extract string literal from an expression (handles StringLiteral and
// potentially other string-like expressions)
static std::string extractStringLiteral(const Expr* E) {
    E = unwrapExpr(E);
    if (const auto* SL = dyn_cast<StringLiteral>(E)) {
        return SL->getString().str();
    }
    return "";
}

// Extract field name from an expression (handles DeclRefExpr, MemberExpr, etc.)
static std::string extractFieldName(const Expr* E) {
    E = unwrapExpr(E);

    // Direct field reference: field_name
    if (const auto* DRE = dyn_cast<DeclRefExpr>(E)) {
        const NamedDecl* ND = dyn_cast<NamedDecl>(DRE->getDecl());
        if (ND && ND->getDeclName().isIdentifier()) {
            // Check if it's a field declaration
            if (isa<FieldDecl>(ND) || isa<VarDecl>(ND)) {
                return ND->getName().str();
            }
        }
    }

    // Member access: this->field_name or obj.field_name
    if (const auto* ME = dyn_cast<MemberExpr>(E)) {
        if (ME->getMemberDecl()->getDeclName().isIdentifier()) {
            if (isa<FieldDecl>(ME->getMemberDecl())) {
                return ME->getMemberDecl()->getName().str();
            }
        }
    }

    return "";
}

// Parse attribute_names: extract string literals from std::forward_as_tuple("name1", "name2", ...)
static std::vector<std::string> parseAttributeNames(const VarDecl* VD) {
    std::vector<std::string> names;
    if (!VD || !VD->hasInit()) {
        return names;
    }

    const Expr* Init = unwrapExpr(VD->getInit());
    // Look for std::forward_as_tuple(...) call
    // Could be: CallExpr with DeclRefExpr, or MemberExpr, or DependentScopeMemberExpr
    if (const auto* CE = dyn_cast<CallExpr>(Init)) {
        std::string calleeName;

        // Try different ways to get the callee name
        const Expr* Callee = unwrapExpr(CE->getCallee());
        if (const auto* DRE = dyn_cast<DeclRefExpr>(Callee)) {
            calleeName = DRE->getDecl()->getNameAsString();
        } else if (const auto* ME = dyn_cast<MemberExpr>(Callee)) {
            if (ME->getMemberDecl()->getDeclName().isIdentifier()) {
                calleeName = ME->getMemberDecl()->getNameAsString();
            }
        }

        if (calleeName == "forward_as_tuple") {
            // Extract all string literal arguments
            for (unsigned i = 0; i < CE->getNumArgs(); ++i) {
                std::string name = extractStringLiteral(CE->getArg(i));
                if (!name.empty()) {
                    names.push_back(name);
                }
            }
        }
    }
    return names;
}

// Helper to find ReturnStmt in a statement (handles CompoundStmt)
static const ReturnStmt* findReturnStmt(const Stmt* S) {
    if (!S) {
        return nullptr;
    }
    if (const auto* RS = dyn_cast<ReturnStmt>(S)) {
        return RS;
    }
    if (const auto* CS = dyn_cast<CompoundStmt>(S)) {
        // Look for ReturnStmt in the compound statement
        for (const auto* SubStmt : CS->body()) {
            if (const auto* RS = findReturnStmt(SubStmt)) {
                return RS;
            }
        }
    }
    // Recursively check children
    for (const Stmt* Child : S->children()) {
        if (const auto* RS = findReturnStmt(Child)) {
            return RS;
        }
    }
    return nullptr;
}

// Parse attribute_values(): extract field references from return std::forward_as_tuple(field1, field2, ...)
static std::vector<std::string> parseAttributeValues(const CXXMethodDecl* MD) {
    std::vector<std::string> values;
    if (!MD || !MD->hasBody()) {
        return values;
    }

    const Stmt* Body = MD->getBody();
    // Look for return statement (may be inside CompoundStmt)
    const ReturnStmt* RS = findReturnStmt(Body);
    if (RS && RS->getRetValue()) {
        const Expr* RetExpr = RS->getRetValue();
        // Look for std::forward_as_tuple(...) call
        if (const CallExpr* CE = findCallExpr(RetExpr)) {
            std::string calleeName;

            // Try different ways to get the callee name
            const Expr* Callee = unwrapExpr(CE->getCallee());
            if (const auto* DRE = dyn_cast<DeclRefExpr>(Callee)) {
                calleeName = DRE->getDecl()->getNameAsString();
            } else if (const auto* ME = dyn_cast<MemberExpr>(Callee)) {
                if (ME->getMemberDecl()->getDeclName().isIdentifier()) {
                    calleeName = ME->getMemberDecl()->getNameAsString();
                }
            }

            if (calleeName == "forward_as_tuple") {
                // Extract all field references
                for (unsigned i = 0; i < CE->getNumArgs(); ++i) {
                    std::string field = extractFieldName(CE->getArg(i));
                    if (!field.empty()) {
                        values.push_back(field);
                    }
                }
            }
        }
    }
    return values;
}

void AttributeProtocolCheck::check(const MatchFinder::MatchResult& Result) {
    const auto* Record = Result.Nodes.getNodeAs<CXXRecordDecl>("target_struct");
    if (!Record) {
        return;
    }

    const SourceManager& SM = *Result.SourceManager;

    // Skip if not in user code (e.g., system headers, third_party)
    if (SM.isInSystemHeader(Record->getLocation())) {
        return;
    }

    // Skip files in build directories or third_party
    StringRef FileName = SM.getFilename(Record->getLocation());
    if (FileName.contains("build") || FileName.contains("third_party") || FileName.contains(".cpmcache")) {
        return;
    }

    // Collect non-static data members
    std::vector<std::string> actualFields;
    for (const auto* Field : Record->fields()) {
        if (!Field->getDeclName().isIdentifier()) {
            continue;
        }
        actualFields.push_back(Field->getName().str());
    }

    if (actualFields.empty()) {
        return;  // Empty struct, nothing to check
    }

    // Find attribute_names and attribute_values() declarations
    const VarDecl* AttributeNamesVD = nullptr;
    const CXXMethodDecl* AttributeValuesMD = nullptr;
    bool hasToHash = false;

    for (const auto* D : Record->decls()) {
        if (const auto* VD = dyn_cast<VarDecl>(D)) {
            if (VD->isStaticDataMember() && VD->getDeclName().isIdentifier() && VD->getName() == "attribute_names") {
                AttributeNamesVD = VD;
            }
        }
        if (const auto* MD = dyn_cast<CXXMethodDecl>(D)) {
            if (MD->getDeclName().isIdentifier()) {
                if (MD->getName() == "attribute_values") {
                    AttributeValuesMD = MD;
                } else if (MD->getName() == "to_hash") {
                    hasToHash = true;
                }
            }
        }
    }

    bool isTensorArgs = Record->getName() == "tensor_args_t";
    std::string structName = Record->getNameAsString();

    // Case 1: Missing protocol entirely
    if (!AttributeNamesVD || !AttributeValuesMD) {
        // For operation_attributes_t with a custom to_hash(), the protocol is optional
        if (!isTensorArgs && hasToHash) {
            return;
        }

        DiagnosticIDs::Level diagLevel = isTensorArgs ? DiagnosticIDs::Error : DiagnosticIDs::Warning;

        auto Diag = diag(
                        Record->getLocation(),
                        "%0 is missing attribute_names/attribute_values() protocol; "
                        "this breaks visit_object_of_type<Tensor> traversal and hashing")
                    << structName;

        // Generate fix-it: insert before closing brace
        SourceLocation RBrace = Record->getBraceRange().getEnd();
        if (!RBrace.isInvalid()) {
            std::ostringstream code;
            code << "\n    static constexpr auto attribute_names = std::forward_as_tuple(";
            for (size_t i = 0; i < actualFields.size(); ++i) {
                if (i > 0) {
                    code << ", ";
                }
                code << "\"" << actualFields[i] << "\"";
            }
            code << ");\n";
            code << "    auto attribute_values() const { return std::forward_as_tuple(";
            for (size_t i = 0; i < actualFields.size(); ++i) {
                if (i > 0) {
                    code << ", ";
                }
                code << actualFields[i];
            }
            code << "); }\n";

            Diag << FixItHint::CreateInsertion(RBrace, code.str());
        }
        return;
    }

    // Case 2: Protocol exists, validate it matches fields
    std::vector<std::string> declaredNames = parseAttributeNames(AttributeNamesVD);
    std::vector<std::string> declaredValues = parseAttributeValues(AttributeValuesMD);

    bool hasMismatch = false;
    std::vector<std::string> missingInNames;
    std::vector<std::string> missingInValues;
    std::vector<std::string> extraInNames;
    std::vector<std::string> extraInValues;

    // Check attribute_names matches fields
    for (const auto& field : actualFields) {
        if (std::find(declaredNames.begin(), declaredNames.end(), field) == declaredNames.end()) {
            missingInNames.push_back(field);
            hasMismatch = true;
        }
    }
    for (const auto& name : declaredNames) {
        if (std::find(actualFields.begin(), actualFields.end(), name) == actualFields.end()) {
            extraInNames.push_back(name);
            hasMismatch = true;
        }
    }

    // Check attribute_values() matches fields
    for (const auto& field : actualFields) {
        if (std::find(declaredValues.begin(), declaredValues.end(), field) == declaredValues.end()) {
            missingInValues.push_back(field);
            hasMismatch = true;
        }
    }
    for (const auto& value : declaredValues) {
        if (std::find(actualFields.begin(), actualFields.end(), value) == actualFields.end()) {
            extraInValues.push_back(value);
            hasMismatch = true;
        }
    }

    // Check order matches
    bool orderMismatch = false;
    if (declaredNames.size() == actualFields.size() && declaredValues.size() == actualFields.size()) {
        for (size_t i = 0; i < actualFields.size(); ++i) {
            if (i < declaredNames.size() && declaredNames[i] != actualFields[i]) {
                orderMismatch = true;
                hasMismatch = true;
                break;
            }
            if (i < declaredValues.size() && declaredValues[i] != actualFields[i]) {
                orderMismatch = true;
                hasMismatch = true;
                break;
            }
        }
    }

    if (!hasMismatch) {
        return;  // Everything matches!
    }

    // Report mismatches
    std::ostringstream msg;
    msg << structName << " has mismatched attribute protocol:";
    if (!missingInNames.empty()) {
        msg << " missing in attribute_names: " << missingInNames[0];
        for (size_t i = 1; i < missingInNames.size(); ++i) {
            msg << ", " << missingInNames[i];
        }
    }
    if (!missingInValues.empty()) {
        msg << " missing in attribute_values(): " << missingInValues[0];
        for (size_t i = 1; i < missingInValues.size(); ++i) {
            msg << ", " << missingInValues[i];
        }
    }
    if (!extraInNames.empty()) {
        msg << " extra in attribute_names: " << extraInNames[0];
        for (size_t i = 1; i < extraInNames.size(); ++i) {
            msg << ", " << extraInNames[i];
        }
    }
    if (!extraInValues.empty()) {
        msg << " extra in attribute_values(): " << extraInValues[0];
        for (size_t i = 1; i < extraInValues.size(); ++i) {
            msg << ", " << extraInValues[i];
        }
    }
    if (orderMismatch) {
        msg << " order mismatch";
    }

    auto Diag = diag(Record->getLocation(), "%0") << msg.str();

    // Generate fix-it: replace the existing attribute_names and attribute_values()
    if (AttributeNamesVD && AttributeValuesMD) {
        SourceRange namesRange = AttributeNamesVD->getSourceRange();
        SourceRange valuesRange = AttributeValuesMD->getSourceRange();

        // Build corrected code
        std::ostringstream namesCode;
        namesCode << "static constexpr auto attribute_names = std::forward_as_tuple(";
        for (size_t i = 0; i < actualFields.size(); ++i) {
            if (i > 0) {
                namesCode << ", ";
            }
            namesCode << "\"" << actualFields[i] << "\"";
        }
        namesCode << ");";

        std::ostringstream valuesCode;
        valuesCode << "auto attribute_values() const { return std::forward_as_tuple(";
        for (size_t i = 0; i < actualFields.size(); ++i) {
            if (i > 0) {
                valuesCode << ", ";
            }
            valuesCode << actualFields[i];
        }
        valuesCode << "); }";

        // Generate fix-its for replacement
        // For attribute_names, replace from the start of the declaration to the semicolon
        if (AttributeNamesVD) {
            SourceLocation namesStart = AttributeNamesVD->getBeginLoc();
            SourceLocation namesEnd = AttributeNamesVD->getEndLoc();
            if (namesStart.isValid() && namesEnd.isValid()) {
                // Extend to include the semicolon - findLocationAfterToken returns location AFTER the token
                const LangOptions& LO = Result.Context->getLangOpts();
                SourceLocation SemiLoc = Lexer::findLocationAfterToken(namesEnd, tok::semi, SM, LO, false);
                if (SemiLoc.isValid()) {
                    // SemiLoc points to after the semicolon, so use it as the end
                    namesEnd = SemiLoc;
                }
                SourceRange fullRange(namesStart, namesEnd);
                Diag << FixItHint::CreateReplacement(fullRange, namesCode.str());
            }
        }

        // For attribute_values(), replace the entire method definition
        if (AttributeValuesMD && AttributeValuesMD->hasBody()) {
            SourceLocation valuesStart = AttributeValuesMD->getBeginLoc();
            SourceLocation valuesEnd = AttributeValuesMD->getEndLoc();
            if (valuesStart.isValid() && valuesEnd.isValid()) {
                // The getEndLoc() should already include the closing brace, but let's make sure
                // we include any trailing whitespace/newline by finding the next token
                const LangOptions& LO = Result.Context->getLangOpts();
                SourceLocation AfterEnd = Lexer::findLocationAfterToken(valuesEnd, tok::unknown, SM, LO, true);
                if (AfterEnd.isValid()) {
                    // Include up to but not including the next token (usually a semicolon or closing brace)
                    valuesEnd = AfterEnd.getLocWithOffset(-1);
                }
                SourceRange fullRange(valuesStart, valuesEnd);
                Diag << FixItHint::CreateReplacement(fullRange, valuesCode.str());
            }
        }
    }
}

}  // namespace tidy
}  // namespace tt
