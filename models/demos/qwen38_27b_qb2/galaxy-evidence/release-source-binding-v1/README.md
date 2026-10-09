# Immutable release source binding, Oct 9 2026 UTC

The TTIS preparer now requires every qualified runtime file and selected
precision policy to exist unchanged in the recorded model commit. Previously,
`git diff --quiet HEAD` ignored untracked and ignored files: a locally qualified
policy could be absent when the image fetched its advertised SHA. The existing
exact G0 source/precision check remains required.

TTIS commit `5f53029442415eeb833b3232cf35a63a8810f1ce` is pushed on
`anatarajan/qwen38-galaxy-release-20261009`. Seventeen tests passed, including
real Git fixtures for missing/changed committed files and preservation of both
precision policies through ModelSpec, the TTIS wrapper and Helm. Re-preparing
the original native bundle with its actual G0 receipt produced byte-identical
manifest, ModelSpec, values, receipt and image verifier files.

The separately pushed model branch
`anatarajan/qwen38-head-control-runtime-20261009` at
`d3e8d6021f7aadcb28ba3903d79bd04b288a2819` adds only the exact queued
`precision_accurate_decode_bfp8_head.json` to the native release source pin.
All 28 runtime/policy hashes match the immutable head-control queue manifest
and exist unchanged at this commit. The three changed policy fields are the
head weight format, head math fidelity and descriptive config ID; all 64
decoder policies remain identical.

This is source preparation, **not a passing head-control G0, GPQA result or new
image**. The experiment remains queued behind the original hardware run.
After it completes, use its actual passing G0 receipt with this model checkout
and `--precision precision_accurate_decode_bfp8_head.json` to prepare a new
bundle. Accuracy, the rebuilt container and hardware/API qualification remain
required. The existing built image and running native endpoint are unchanged.

Evidence: [source comparison](head-source.json),
[17-test JUnit report](commit-source-tests.xml), and
[release instructions](https://github.com/tenstorrent/tt-inference-server/blob/5f53029442415eeb833b3232cf35a63a8810f1ce/scripts/release/QWEN38_GALAXY.md).
