"""The `optimize` startup block prints a repeated absolute demo path and a redundant
`[optimize/cc]` prefix on every one-time startup line (isolation, dashboard, --persist,
--fresh) that already sits directly under the boxed header naming the run. Neither
adds information a human hasn't already seen two lines above; both just make the log
noisier. Pin the fix as source-text assertions -- cmd_optimize has too many
subprocess/device side effects to unit-test end-to-end, matching the convention already
used for print-format checks elsewhere in this tool.
"""

from __future__ import annotations

from pathlib import Path

_SRC = (Path(__file__).resolve().parents[1] / "commands" / "optimize.py").read_text(encoding="utf-8")


def test_the_header_model_line_prefers_a_path_relative_to_repo_root():
    assert "_demo_disp = demo_dir.relative_to(repo_root)" in _SRC
    assert 'print(f"  model    : {_demo_disp} ({kind})")' in _SRC


def test_startup_lines_no_longer_repeat_the_optimize_cc_prefix():
    for _gone in (
        '"  [optimize/cc] existing demo -> isolated on branch',
        '"  [optimize/cc] dashboard:',
        '"  [optimize/cc] --persist:',
        '"  [optimize/cc] --fresh:',
    ):
        assert _gone not in _SRC


def test_startup_lines_keep_an_aligned_label_instead():
    for _kept in ("  isolation :", "  dashboard :", "  --persist :", "  --fresh   :"):
        assert _kept in _SRC
