"""Review round 10, hard-check gap: the bf16-DEST candidate for the SDPA forward as intended (8-tile DEST: kPass = 8,
one DEST pass per 8-row chunk in QK^T and exp; fp32_dest_acc_en = false, the fp32 CBs read through SrcA).
Experiment only: apply, build, measure, then `git checkout -- <factory>`."""
import sys

F = "tt-train/sources/ttml/metal/ops/sdpa_fw/device/sdpa_fw_program_factory.cpp"
s = open(F).read()
for old, new, n in [
    (
        "    constexpr bool fp32_dest_acc_en = true;\n",
        "    constexpr bool fp32_dest_acc_en = false;  // EXPERIMENT (review round 10): bf16 DEST, 8 tiles per pass\n",
        1,
    ),
    (".fp32_dest_acc_en = true,", ".fp32_dest_acc_en = false,", 3),
    (
        "        mode[kPrevMmOutCbIndex] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;\n        mode[kCurMmOutCbIndex] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;\n",
        "        mode[kPrevMmOutCbIndex] = tt::tt_metal::UnpackToDestMode::Default;  // EXPERIMENT\n        mode[kCurMmOutCbIndex] = tt::tt_metal::UnpackToDestMode::Default;\n",
        1,
    ),
]:
    if s.count(old) != n:
        print("FAILED", old[:60], s.count(old))
        sys.exit(2)
    s = s.replace(old, new)
open(F, "w").write(s)
print("bf16 DEST experiment applied (dst_size 8 -> kPass 8)")
