# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Build an isolated reader-source override; never edit the installed runtime."""

import hashlib
import json
from pathlib import Path

READER = Path("ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/reader_decode_all.cpp")
COMMON = READER.with_name("dataflow_common.hpp")
FACTORY = READER.parents[2] / "sdpa_decode_program_factory.cpp"
PINNED = {
    str(READER): "2a70c7a0b32f8972efc7d15cbfb735e493838121862ef9faf8fe1ba97845e591",
    str(COMMON): "cacb9efbd62334f8abc371b9c0da5ab54bcd867fa06e0e723a65e76a53496d89",
    str(FACTORY): "f770ccf8ffad855aac3ac5063b43cd4cfab8fd732d3adb512dbee5051d0eda69",
}
VARIANTS = ("native", "kv4", "kv8", "kv16")
CASES = ((32768, 16), (131072, 8), (262016, 4), (32768, 32), (131072, 16), (262016, 8))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError("Reader source anchor is missing or ambiguous")
    return text.replace(before, after, 1)


def render_reader(text, common, variant):
    if variant not in VARIANTS:
        raise ValueError("Unsupported reader variant")
    common = str(common)
    if any(char in common for char in ('"', "\n", "\r", "\\")) or not Path(common).is_absolute():
        raise ValueError("Common header must have a safe absolute path")
    text = replace_once(text, '#include "dataflow_common.hpp"', f'#include "{common}"')
    if variant == "native":
        return text
    limit = int(variant[2:])
    anchor = "    uint32_t barrier_count = 0;"
    addition = f"""    // Isolated Qwen reader experiment: retain Q/mask throttling and final
    // K/V completion barriers; vary only the intermediate paged-KV barriers.
    constexpr bool qwen_reader_geometry = is_paged_attention && is_causal &&
        num_kv_heads == 1 && DHt == 8 && vDHt == 8 && PNHt == 1 &&
        Sk_chunk_t == 8 && !use_half_tile && !reuse_k && !use_k_mcast &&
        !use_attention_mask && !spec_multi_pos &&
        q_tile_bytes == 2048 && k_tile_bytes == 1088 && v_tile_bytes == 1088;
    static_assert(qwen_reader_geometry, "Reader diagnostic requires the fixed Qwen attention geometry");
    constexpr uint32_t kv_barrier_threshold = {limit};
{anchor}"""
    text = replace_once(text, anchor, addition)
    for tensor in ("k", "v"):
        before = f"                    {tensor}_tile_bytes,\n                    barrier_threshold,"
        text = replace_once(text, before, before.replace("barrier_threshold", "kv_barrier_threshold"))
    return text


def build_overlay(native, destination, variant):
    native, destination = native.resolve(), destination.resolve()
    for relative, expected in PINNED.items():
        if sha(native / relative) != expected:
            raise ValueError(f"Pinned reader dependency changed: {relative}")
    reader = render_reader((native / READER).read_text(), native / COMMON, variant)
    destination.mkdir(parents=True, exist_ok=False)
    target = destination / READER
    target.parent.mkdir(parents=True)
    target.write_text(reader)
    manifest = dict(
        variant=variant,
        overlay=str(destination),
        native=str(native),
        reader=str(target),
        reader_sha256=sha(target),
        native_sha256=PINNED,
        precision_change=False,
        changes="Only paged-KV intermediate read-barrier frequency; final barriers and buffer ownership unchanged",
    )
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_overlay(manifest, kernel_path, cwd):
    if Path(kernel_path).resolve() != Path(manifest["overlay"]):
        raise ValueError("Reader override path does not match the manifest")
    if (Path(cwd) / READER).exists():
        raise ValueError("Working directory shadows the reader override")
    if sha(Path(manifest["reader"])) != manifest["reader_sha256"]:
        raise ValueError("Reader override changed")
    for relative, expected in manifest["native_sha256"].items():
        if sha(Path(manifest["native"]) / relative) != expected:
            raise ValueError(f"Native dependency changed: {relative}")


def compilation_evidence(cache, manifest):
    """Require JIT-generated includes to name this exact override, not just env."""
    evidence = []
    for path in cache.rglob("kernel_includes.hpp"):
        text = path.read_text()
        if READER.name not in text:
            continue
        if f'#include "{manifest["reader"]}"' not in text:
            raise ValueError("JIT compiled a different attention reader")
        evidence.append(dict(path=str(path), sha256=sha(path)))
    if not evidence:
        raise ValueError("No JIT evidence for the reader override")
    return evidence


def compare_readers(reports):
    if len(reports) != 5 or [r["variant"] for r in reports] != [*VARIANTS, "native"]:
        raise ValueError("Need native, all candidate variants, then a repeated native control")
    if any(r.get("passed") is not True or r.get("cleanup_completed") is not True for r in reports):
        raise ValueError("Reader comparisons require passing clean-device receipts")
    comparisons = []
    for index, (length, batch) in enumerate(CASES):
        rows = [r["cases"][index] for r in reports]
        if any((row["input_tokens"], row["batch"]) != (length, batch) for row in rows):
            raise ValueError("Reader comparison geometry mismatch")
        timings = [row["candidates"][0]["median_traced_call_us"] for row in rows]
        drift = abs(timings[-1] / timings[0] - 1)
        valid = [i for i in range(4) if rows[i]["passed"] and rows[i]["selection"]["timing_comparison_qualified"]]
        if not valid:
            raise ValueError("No internally stable, accurate reader measurement")
        best = min(valid, key=lambda i: timings[i])
        comparisons.append(
            dict(
                input_tokens=length,
                batch=batch,
                baseline_repeat_drift_fraction=drift,
                timing_comparison_qualified=drift <= 0.03
                and rows[0]["selection"]["timing_comparison_qualified"]
                and rows[-1]["selection"]["timing_comparison_qualified"],
                fastest_passing_variant=VARIANTS[best],
                baseline_over_fastest=(timings[0] + timings[-1]) / (2 * timings[best]),
                timings_us=dict(zip([*VARIANTS, "native_repeat"], timings)),
            )
        )
    return comparisons
