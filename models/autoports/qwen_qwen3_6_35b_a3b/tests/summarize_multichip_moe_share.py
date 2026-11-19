# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only MoE kernel-work and device-firmware spans from decode profiles.

Kernel-work sums and within-replay firmware envelopes are separate measurements;
neither is a fraction of clean host elapsed latency or exclusive causal time.
The source boundary includes router, selected/shared experts and output CCL;
post-attention normalization and the final residual addition are excluded.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

DOC = Path(__file__).parents[1] / "doc/multichip_decoder"


def _selector_boundary(rows, *, batch, decode_fp32_selector):
    topk = [i for i, row in enumerate(rows) if row["OP CODE"] == "TopKDeviceOperation"]
    # generic_op records the selector's preallocated values/IDs as inputs and
    # its final argument (IDs) as output. Inline source alone is not an identity:
    # require the complete 256-logit -> 8-value/ID singleton signature as well.
    signature = {
        "OP CODE": "GenericOpDeviceOperation",
        "INPUT_0_W_PAD[LOGICAL]": "1[1]",
        "INPUT_0_Z_PAD[LOGICAL]": "1[1]",
        "INPUT_0_Y_PAD[LOGICAL]": "32[1]",
        "INPUT_0_X_PAD[LOGICAL]": "256[256]",
        "INPUT_0_DATATYPE": "FLOAT32",
        "INPUT_1_Y_PAD[LOGICAL]": "32[1]",
        "INPUT_1_X_PAD[LOGICAL]": "32[8]",
        "INPUT_1_DATATYPE": "FLOAT32",
        "INPUT_2_Y_PAD[LOGICAL]": "32[1]",
        "INPUT_2_X_PAD[LOGICAL]": "32[8]",
        "INPUT_2_DATATYPE": "UINT32",
        "OUTPUT_0_Y_PAD[LOGICAL]": "32[1]",
        "OUTPUT_0_X_PAD[LOGICAL]": "32[8]",
        "OUTPUT_0_DATATYPE": "UINT32",
        "COMPUTE KERNEL SOURCE": "[]",
        "DATA MOVEMENT KERNEL SOURCE": "['Inline source']",
    }
    fp32 = [i for i, row in enumerate(rows) if all(row.get(key) == value for key, value in signature.items())]
    if batch == 1 and decode_fp32_selector:
        assert len(fp32) == 1 and not topk, f"ambiguous/missing policy-selected FP32 selector: {fp32=}, {topk=}"
        return fp32[0], "MeshFP32TopK"
    assert len(topk) == 1 and not fp32, f"ambiguous/missing native TopK boundary: {topk=}, {fp32=}"
    return topk[0], "TopKDeviceOperation"


def _firmware_span(rows):
    starts = [int(row["DEVICE FW START CYCLE"]) for row in rows]
    ends = [int(row["DEVICE FW END CYCLE"]) for row in rows]
    assert all(0 <= start < end for start, end in zip(starts, ends)), "invalid firmware timestamps"
    return max(ends) - min(starts)


def measure(path, *, batch, split_rows_once=False, decode_fp32_selector=False):
    sessions = defaultdict(lambda: defaultdict(list))
    active = False
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["OP TYPE"] == "signpost":
                if row["OP CODE"] == "PERF_DECODE":
                    assert not active
                    active = True
                elif row["OP CODE"] == "PERF_DECODE_END":
                    break
            elif active and row["OP TYPE"] == "tt_dnn_device":
                sessions[row["DEVICE ID"]][row["METAL TRACE REPLAY SESSION ID"]].append(
                    {
                        key: row[key]
                        for key in (
                            "OP CODE",
                            "DEVICE KERNEL DURATION [ns]",
                            "DEVICE FW START CYCLE",
                            "DEVICE FW END CYCLE",
                            "INPUT_0_X_PAD[LOGICAL]",
                            "INPUT_1_X_PAD[LOGICAL]",
                            "INPUT_1_DATATYPE",
                            "INPUT_0_W_PAD[LOGICAL]",
                            "INPUT_0_Z_PAD[LOGICAL]",
                            "INPUT_0_Y_PAD[LOGICAL]",
                            "INPUT_0_DATATYPE",
                            "INPUT_1_Y_PAD[LOGICAL]",
                            "INPUT_2_Y_PAD[LOGICAL]",
                            "INPUT_2_X_PAD[LOGICAL]",
                            "INPUT_2_DATATYPE",
                            "OUTPUT_0_Y_PAD[LOGICAL]",
                            "OUTPUT_0_X_PAD[LOGICAL]",
                            "OUTPUT_0_DATATYPE",
                            "COMPUTE KERNEL SOURCE",
                            "DATA MOVEMENT KERNEL SOURCE",
                        )
                    }
                )
    assert set(sessions) == {"0", "1", "2", "3"}
    ranks = {}
    for rank, replays in sessions.items():
        assert len(replays) == 16
        totals, moe, split, bounds, selectors = [], [], [], [], []
        layer_spans, moe_spans = [], []
        for rows in replays.values():
            selector, selector_kind = _selector_boundary(rows, batch=batch, decode_fp32_selector=decode_fp32_selector)
            selectors.append((selector, selector_kind))
            routers = [
                i
                for i, row in enumerate(rows[:selector])
                if (
                    row["OP CODE"] == "MatmulDeviceOperation"
                    and row["INPUT_1_DATATYPE"] == "FLOAT32"
                    and row["INPUT_1_X_PAD[LOGICAL]"] == "256[256]"
                )
                or (row["OP CODE"] == "GenericOpDeviceOperation" and row["INPUT_0_X_PAD[LOGICAL]"] == "256[256]")
            ]
            assert routers, "router boundary absent"
            start = routers[-1]
            if start > 0 and rows[start - 1]["OP CODE"] == "ShardedToInterleavedDeviceOperation":
                start -= 1
            stop = max(i for i, row in enumerate(rows) if row["OP CODE"] == "BinaryNgDeviceOperation")
            assert start < selector < stop
            tail = [row["OP CODE"] for row in rows[stop + 1 :]]
            # The replicated batched direct-AR residual can require one final
            # L1-sharded -> DRAM conversion before the public row splitter.
            if tail and tail[0] == "ShardedToInterleavedDeviceOperation":
                tail = tail[1:]
            if batch > 1 and split_rows_once:
                assert (
                    tail
                    == ["UntilizeCodegenDeviceOperation"]
                    + [
                        "SliceDeviceOperation",
                        "TilizeWithValPaddingDeviceOperation",
                    ]
                    * batch
                ), f"unexpected single-untilize output split: {tail}"
            elif "UntilizeCodegenDeviceOperation" in tail:
                # decode_batch returns 32 one-row tensors. Row 0 is a tiled
                # slice; the remaining unaligned rows lower to this triplet.
                assert tail == ["SliceDeviceOperation"] + [
                    "UntilizeCodegenDeviceOperation",
                    "SliceDeviceOperation",
                    "TilizeWithValPaddingDeviceOperation",
                ] * (batch - 1)
            else:
                assert all(code == "SliceDeviceOperation" for code in tail)
            assert any("AllReduce" in row["OP CODE"] or "AllGather" in row["OP CODE"] for row in rows[start:stop])
            durations = [float(row["DEVICE KERNEL DURATION [ns]"]) / 1e6 for row in rows]
            assert all(value > 0 for value in durations)
            totals.append(sum(durations))
            moe.append(sum(durations[start:stop]))
            split.append(sum(durations[stop + 1 :]))
            bounds.append((start, stop, len(rows)))
            layer_span, moe_span = _firmware_span(rows), _firmware_span(rows[start:stop])
            assert 0 < moe_span <= layer_span
            layer_spans.append(layer_span)
            moe_spans.append(moe_span)
        assert len(set(bounds)) == 1
        assert len(set(selectors)) == 1
        ranks[rank] = dict(
            whole_layer_kernel_sum_ms=sum(totals) / 16,
            moe_kernel_sum_ms=sum(moe) / 16,
            moe_kernel_work_fraction=sum(moe) / sum(totals),
            output_conversion_kernel_sum_ms=sum(split) / 16,
            output_split="single_untilize" if batch > 1 and split_rows_once else "legacy",
            operation_range_start_inclusive=bounds[0][0],
            operation_range_stop_exclusive=bounds[0][1],
            layer_operation_count=bounds[0][2],
            replay_count=16,
            selector_operation_index=selectors[0][0],
            selector_kind=selectors[0][1],
            whole_layer_device_fw_span_cycles=sum(layer_spans) / 16,
            moe_device_fw_span_cycles=sum(moe_spans) / 16,
            moe_device_fw_span_fraction=sum(moe_spans) / sum(layer_spans),
        )
    critical = max(ranks, key=lambda rank: ranks[rank]["whole_layer_kernel_sum_ms"])
    return dict(critical_rank=critical, ranks=ranks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", type=Path, default=DOC)
    args = parser.parse_args()
    rows = []
    for batch, prefix in ((1, ""), (32, "perf_b32")):
        for layer in (0, 3):
            relative = Path(prefix) / f"tracy/layer_{layer}/ops.csv"
            correctness = json.loads((args.doc / relative.parent / "correctness.json").read_text())
            assert correctness["passed"]
            policy = correctness["multichip"]["precision_policy"]
            rows.append(
                dict(
                    batch=batch,
                    layer=layer,
                    source=str(relative),
                    **measure(
                        args.doc / relative,
                        batch=batch,
                        split_rows_once=policy.get("split_rows_once", False),
                        decode_fp32_selector=policy.get("decode_fp32_selector", False),
                    ),
                )
            )
    result = dict(
        scope="per-rank kernel-work fraction and within-replay device firmware envelope fraction; neither is a fraction of clean host elapsed decode latency",
        boundary="router input conversion through routed/shared expert combination and output collective; excludes normalization/final residual/output split",
        device_span_definition="Within each device/replay, max(DEVICE FW END CYCLE) minus min(DEVICE FW START CYCLE), for the MoE operation range and whole layer including output split. Fraction is sum(MoE spans)/sum(layer spans) across16 replays; clock scale cancels. Envelopes include intervening gaps and may overlap other work; not exclusive causal attribution.",
        timestamp_limit="ops.csv exposes firmware start/end cycles, not absolute kernel start/end cycles. DEVICE KERNEL DURATION alone cannot reconstruct kernel envelopes. Firmware envelopes are reported explicitly instead; no cross-device clock subtraction or clean-latency division.",
        measurements=rows,
    )
    (args.doc / "moe_share.json").write_text(json.dumps(result, indent=2) + "\n")
    for row in rows:
        values = row["ranks"][row["critical_rank"]]
        print(
            f"layer {row['layer']} batch {row['batch']}: MoE kernel-work share {values['moe_kernel_work_fraction']:.2%}; "
            f"within-replay device firmware span share {values['moe_device_fw_span_fraction']:.2%}"
        )


if __name__ == "__main__":
    main()
