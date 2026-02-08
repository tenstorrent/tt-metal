// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "AttributeProtocolCheck.h"

#include <clang-tidy/ClangTidyModule.h>
#include <clang-tidy/ClangTidyModuleRegistry.h>

namespace tt {
namespace tidy {

class TTModule : public clang::tidy::ClangTidyModule {
public:
    void addCheckFactories(clang::tidy::ClangTidyCheckFactories& CheckFactories) override {
        CheckFactories.registerCheck<AttributeProtocolCheck>("tt-attribute-protocol");
    }
};

}  // namespace tidy
}  // namespace tt

// Register this module with clang-tidy
static clang::tidy::ClangTidyModuleRegistry::Add<tt::tidy::TTModule> X(
    "tt-module", "Tenstorrent custom clang-tidy checks");

// This anchor is used to force the linker to link in the generated object file
// and thus register the module.
volatile int TTModuleAnchorSource = 0;
