# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared summary tables for Voxtral ISL perf sweeps."""

from __future__ import annotations

from loguru import logger


def log_e2e_isl_sweep_summary(rows: list[dict]) -> None:
    """Print a demo-aligned full-pipeline ISL sweep table."""
    if not rows:
        return

    header = (
        f"{'label':<16} {'seq':>6} {'chars':>6} {'lat_ms':>9} {'ttfa_ms':>9} "
        f"{'audio_s':>8} {'rtf':>7} {'char/s':>8} {'rt_x':>6} {'frames':>7} {'end':>4}"
    )
    lines = [
        "",
        "=" * len(header),
        "Voxtral full-pipeline ISL sweep summary (RTF = latency/audio; rt_x = audio/latency)",
        "=" * len(header),
        header,
        "-" * len(header),
    ]
    for row in rows:
        latency_s = float(row.get("latency_s", 0.0))
        audio_s = float(row.get("audio_duration_s", 0.0))
        rtf = float(row.get("rtf", 0.0))
        rt_x = audio_s / latency_s if latency_s > 0 else 0.0
        lines.append(
            f"{str(row.get('label', '')):<16} "
            f"{int(row.get('prompt_seq_len', 0)):>6} "
            f"{int(row.get('n_chars', 0)):>6} "
            f"{float(row.get('latency_ms', 0.0)):>9.1f} "
            f"{float(row.get('ttfa_ms', 0.0)):>9.1f} "
            f"{audio_s:>8.2f} "
            f"{rtf:>7.3f} "
            f"{float(row.get('chars_per_s', 0.0)):>8.1f} "
            f"{rt_x:>6.2f} "
            f"{int(row.get('n_frames', 0)):>7} "
            f"{'Y' if row.get('hit_end_audio') else 'N':>4}"
        )
    lines.append("=" * len(header))
    logger.info("\n".join(lines))


def log_text_prefill_isl_sweep_summary(rows: list[dict]) -> None:
    """Print text-backbone-only prefill ISL sweep table."""
    if not rows:
        return

    header = f"{'isl':>6} {'mode':<10} {'paged':>5} {'prefill_s':>10} {'ms/tok':>8} " f"{'tok/s':>8} {'decode_ms':>10}"
    lines = [
        "",
        "=" * len(header),
        "Voxtral text-backbone ISL prefill sweep summary",
        "=" * len(header),
        header,
        "-" * len(header),
    ]
    for row in rows:
        lines.append(
            f"{int(row.get('isl', 0)):>6} "
            f"{str(row.get('prefill_mode', '')):<10} "
            f"{'Y' if row.get('paged_kv') else 'N':>5} "
            f"{float(row.get('prefill_total_s', 0.0)):>10.3f} "
            f"{float(row.get('prefill_ms_per_token', 0.0)):>8.3f} "
            f"{float(row.get('prefill_tokens_per_s', 0.0)):>8.1f} "
            f"{float(row.get('decode_ms', 0.0)):>10.3f}"
        )
    lines.append("=" * len(header))
    logger.info("\n".join(lines))
