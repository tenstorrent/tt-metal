# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from models.demos.qwen38_27b_qb2.tests.gdn_phase_profile import instrument
from models.demos.qwen38_27b_qb2.tt.gdn_step.op import kernel_source


def test_instrumentation_preserves_cpp_tokens_except_scopes_and_markers():
    import re

    for name in ("reader.cpp", "writer.cpp", "compute.cpp"):
        original = kernel_source(name)
        annotated = instrument(name, original)
        restored = annotated.removeprefix('#include "tools/profiler/kernel_profiler.hpp"\n')
        restored = re.sub(r'\{\nDeviceZoneScopedN\("GDN_[A-Z0-9_]+"\);\n', "", restored)
        # Removing all braces here verifies token preservation while allowing
        # the extra local scopes; hardware compilation checks their lifetime.
        tokens = lambda text: re.sub(r"[{}\s]", "", text)
        assert tokens(restored) == tokens(original)


def test_changed_anchor_fails_closed(expect_error):
    with expect_error(ValueError, "anchor changed"):
        instrument("reader.cpp", kernel_source("reader.cpp").replace("cb_reserve_back(0, 4)", "cb_reserve_back(0, 8)"))
