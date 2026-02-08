// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <clang-tidy/ClangTidyCheck.h>

namespace tt {
namespace tidy {

/// Clang-tidy check that enforces the attribute_names / attribute_values() protocol.
///
/// The check fires on any struct named `tensor_args_t` or `operation_attributes_t`
/// (the two struct types used by the device_operation framework) that is missing
/// either `attribute_names` or `attribute_values()`.
///
/// When run with --fix, it generates both members automatically from the struct's
/// non-static data members.
class AttributeProtocolCheck : public clang::tidy::ClangTidyCheck {
public:
    AttributeProtocolCheck(llvm::StringRef Name, clang::tidy::ClangTidyContext* Context);

    void registerMatchers(clang::ast_matchers::MatchFinder* Finder) override;
    void check(const clang::ast_matchers::MatchFinder::MatchResult& Result) override;

    bool isLanguageVersionSupported(const clang::LangOptions& LangOpts) const override { return LangOpts.CPlusPlus17; }
};

}  // namespace tidy
}  // namespace tt
