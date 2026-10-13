# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Collect and format tt-telemetry metrics from the local Prometheus endpoint."""

from __future__ import annotations

import logging
from pathlib import Path

import requests
from prometheus_client.parser import text_string_to_metric_families

log = logging.getLogger(__name__)


SLURM_TELEMETRY_PORT = 8080
ORCHESTRATION_TELEMETRY_PORT = 18080


def telemetry_port_for_launch_mode(launch_mode: str, override: int | None = None) -> int:
    """Return the Prometheus endpoint port to scrape.

    Each deployment has its own default (tt-telemetry listens on a different
    port under Kubernetes to avoid host-port clashes), but ``override`` — the
    runner's ``--telemetry-port`` — wins so a deployment that moves the endpoint
    doesn't need a code change.
    """
    if override is not None:
        return override
    return ORCHESTRATION_TELEMETRY_PORT if launch_mode == "orchestration" else SLURM_TELEMETRY_PORT


TELEMETRY_METRICS = frozenset(
    {
        "tt_ai_clock_mhz",
        "tt_ai_clock_limit_mhz",
        "tt_cable_present",
        "tt_chip_count",
        "tt_dram_trained",
        "tt_eth_firmware_signature",
        "tt_ethernet_cable_present",
        "tt_ethernet_corrected_codeword_count",
        "tt_ethernet_crc_error_count",
        "tt_ethernet_heartbeat",
        "tt_ethernet_link_up",
        "tt_ethernet_retrain_count",
        "tt_ethernet_uncorrected_codeword_count",
        "tt_noc_alive",
        "tt_pcie_link_alive",
    }
)

_STATUS_METRICS = frozenset(
    {
        "tt_cable_present",
        "tt_dram_trained",
        "tt_ethernet_cable_present",
        "tt_ethernet_heartbeat",
        "tt_ethernet_link_up",
        "tt_noc_alive",
        "tt_pcie_link_alive",
    }
)

_COUNTER_METRICS = frozenset(
    {
        "tt_ethernet_corrected_codeword_count",
        "tt_ethernet_crc_error_count",
        "tt_ethernet_retrain_count",
        "tt_ethernet_uncorrected_codeword_count",
    }
)

_VALUE_METRICS = frozenset(
    {
        "tt_ai_clock_limit_mhz",
        "tt_ai_clock_mhz",
        "tt_chip_count",
    }
)

# Firmware reports a field it cannot read as all ones, so the marker is the
# field's own width and has to be matched per metric: EthernetMetrics in
# physical_system_descriptor.hpp makes the retrain and CRC counters uint32 and
# the codeword counters uint64. Matching the wrong one discards live data, since
# a uint64 counter reaches 0xFFFFFFFF legitimately. By the same rule the narrow
# markers are absent: 255 and 65535 are counts a busy link genuinely reaches.
_U32_UNREADABLE = frozenset({0xFFFFFFFF})
# float64 cannot represent the 64-bit marker and rounds it up by one, so both
# forms are listed; whether a sample arrives as int or float is the parser's call.
_U64_UNREADABLE = frozenset({0xFFFFFFFFFFFFFFFF, float(0xFFFFFFFFFFFFFFFF)})

_UNREADABLE_VALUES = {
    # Not a width, but no clock reaches 0xFFFFFFFF MHz, so the marker is safe.
    "tt_ai_clock_mhz": _U32_UNREADABLE,
    "tt_ethernet_corrected_codeword_count": _U64_UNREADABLE,
    "tt_ethernet_crc_error_count": _U32_UNREADABLE,
    "tt_ethernet_retrain_count": _U32_UNREADABLE,
    "tt_ethernet_uncorrected_codeword_count": _U64_UNREADABLE,
}


def _split_unreadable(name: str, samples: list[dict]) -> tuple[list[dict], list[dict]]:
    """Partition samples into ones the chip could report and ones it could not.

    A metric with no marker of its own keeps every sample, so a new one is
    reported as collected until its width is added above.
    """
    markers = _UNREADABLE_VALUES.get(name, frozenset())
    readable = [s for s in samples if s["value"] not in markers]
    unreadable = [s for s in samples if s["value"] in markers]
    return readable, unreadable


def _write_dump(dump_path: Path, body: bytes) -> None:
    """Keep the endpoint's reply on disk, as the bytes it arrived as.

    ``resp.text`` would decode with whatever encoding requests guessed and
    ``write_text`` re-encode with the locale's, which loses exactly the
    malformed reply the dump is most wanted for. An unwritable results_dir is
    logged rather than raised: this is an artifact, not the run's verdict.
    """
    try:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_bytes(body)
    except OSError as exc:
        log.warning("Could not write raw telemetry to %s: %s", dump_path, exc)
    else:
        log.info("Raw telemetry (%d bytes) written to %s", len(body), dump_path)


def collect_prometheus_metrics(
    port: int = SLURM_TELEMETRY_PORT, dump_path: Path | None = None
) -> dict[str, list[dict]] | None:
    """Collect telemetry metrics from the local Prometheus endpoint.

    Returns a dict mapping metric name to a list of
    ``{"labels": {…}, "value": float}`` dicts, or *None* if the endpoint is
    unreachable or no relevant metrics are found.

    ``dump_path`` keeps the reply verbatim, because only TELEMETRY_METRICS is
    parsed out of it and that is a sixth of what the endpoint sends. The rest is
    gone the moment this returns: per-link queue drops and resends, the
    ``tt_fabric_*`` bandwidth, packet and router-state families, and the
    per-device readable percentages. Written before parsing so a reply the parser
    chokes on is still on disk to look at.
    """
    url = f"http://localhost:{port}/metrics"

    if dump_path is not None:
        # Dropped before the request, not after a failed one: the requeue from a
        # self-heal reboot keeps the job id and so reuses results_dir, and a
        # scrape taken just after that reboot is the likeliest one to fail. The
        # previous run's dump would then be attached as this run's telemetry.
        dump_path.unlink(missing_ok=True)

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.info("Prometheus metrics endpoint not available at %s: %s", url, exc)
        return None

    if dump_path is not None:
        _write_dump(dump_path, resp.content)

    metrics: dict[str, list[dict]] = {}
    for family in text_string_to_metric_families(resp.text):
        if family.name not in TELEMETRY_METRICS:
            continue
        for sample in family.samples:
            metrics.setdefault(family.name, []).append({"labels": dict(sample.labels), "value": sample.value})

    return metrics if metrics else None


def _sample_ident(labels: dict[str, str]) -> str:
    """Build a short human-readable identifier from sample labels."""
    parts = [
        f"tray={labels['tray']}" if "tray" in labels else None,
        f"chip={labels['chip']}" if "chip" in labels else None,
        f"ch={labels['channel']}" if "channel" in labels else None,
        f"port={labels['port_id']}" if "port_id" in labels else None,
    ]
    ident = " ".join(p for p in parts if p)
    remote = labels.get("remote_hostname")
    if remote:
        ident += f" -> {remote}"
    return ident


def format_prometheus_metrics(metrics: dict[str, list[dict]]) -> str:
    """Return a human-readable summary of the collected telemetry metrics."""
    lines = ["--- prometheus telemetry metrics ---"]

    for name in sorted(metrics):
        samples = metrics[name]
        lines.append(f"\n  {name}: {len(samples)} samples")

        if name in _STATUS_METRICS:
            up = sum(1 for s in samples if s["value"] == 1)
            down = sum(1 for s in samples if s["value"] == 0)
            lines.append(f"    up/present={up}  down/absent={down}")
            for s in samples:
                if s["value"] == 0:
                    lines.append(f"    DOWN: {_sample_ident(s['labels'])}")

        elif name in _COUNTER_METRICS:
            readable, unreadable = _split_unreadable(name, samples)
            if unreadable:
                lines.append(f"    unreadable={len(unreadable)} (excluded from total)")
                for s in unreadable:
                    lines.append(f"    UNREADABLE: {_sample_ident(s['labels'])}")
            nonzero = [s for s in readable if s["value"] > 0]
            lines.append(f"    non-zero={len(nonzero)}/{len(readable)}")
            if nonzero:
                total = sum(s["value"] for s in nonzero)
                max_val = max(s["value"] for s in nonzero)
                lines.append(f"    total={int(total)}  max={int(max_val)}")
                if name in (
                    "tt_ethernet_uncorrected_codeword_count",
                    "tt_ethernet_crc_error_count",
                ):
                    for s in sorted(nonzero, key=lambda x: x["value"], reverse=True)[:5]:
                        lines.append(f"    {_sample_ident(s['labels'])}: {int(s['value'])}")

        elif name == "tt_eth_firmware_signature":
            unique_sigs = sorted(set(int(s["value"]) for s in samples))
            lines.append(f"    unique signatures: {', '.join(hex(s) for s in unique_sigs)}")

        elif name in _VALUE_METRICS:
            for s in samples:
                ident = _sample_ident(s["labels"])
                if not ident:
                    extra = {k: v for k, v in s["labels"].items() if k not in ("hostname", "__name__")}
                    ident = " ".join(f"{k}={v}" for k, v in sorted(extra.items()))
                suffix = f" [{ident}]" if ident else ""
                lines.append(f"    {name}={int(s['value'])}{suffix}")

    lines.append("\n--- end prometheus metrics ---")
    return "\n".join(lines)


def aggregate_telemetry_for_csv(metrics: dict[str, list[dict]] | None) -> dict:
    """Reduce raw metric families into the flat summary runs.csv expects.

    ``format_prometheus_metrics`` only builds the human-readable log block; the
    CSV verdict needs a separate flat dict whose keys mirror the ones consumed by
    ``analyze_health_check_results.runs_row`` (``available``, the minimum AI clock
    and the summed Ethernet error totals). Returns ``{"available": False}`` when
    nothing was collected so the run row records ``telemetry_available=0`` rather
    than silently claiming a value. A metric that wasn't collected stays *None*
    so its column is blank instead of a fabricated 0.

    Samples a chip could not report are dropped (see ``_UNREADABLE_VALUES``), so
    a total covers the links that answered and ``unreadable_samples`` counts the
    rest.
    """
    if not metrics:
        return {"available": False}

    unreadable: list[dict] = []

    def _values(name: str) -> list[float]:
        readable, dropped = _split_unreadable(name, metrics.get(name, []))
        unreadable.extend(dropped)
        return [s["value"] for s in readable]

    aiclk = _values("tt_ai_clock_mhz")
    retrain = _values("tt_ethernet_retrain_count")
    crc = _values("tt_ethernet_crc_error_count")
    uncorr = _values("tt_ethernet_uncorrected_codeword_count")

    if unreadable:
        log.warning(
            "%d telemetry sample(s) reported as unavailable and left out of the totals: %s",
            len(unreadable),
            ", ".join(sorted({_sample_ident(s["labels"]) for s in unreadable})),
        )

    return {
        "available": True,
        "min_aiclk_mhz": min(aiclk) if aiclk else None,
        "eth_retrain_total": int(sum(retrain)) if retrain else None,
        "eth_crc_total": int(sum(crc)) if crc else None,
        "eth_uncorr_cw_total": int(sum(uncorr)) if uncorr else None,
        "unreadable_samples": len(unreadable),
    }
