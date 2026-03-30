#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SLURM Training Service — bridges tt-dashboard to real TT hardware via SLURM.

Implements the same REST API as the mock-training-service (Go), but dispatches
real sbatch jobs using job_manager.py and streams back actual metrics/logs from
the training output files.

Usage on a login node with SLURM access:
    pip install -r requirements_streamlit.txt  # already has pyyaml, requests
    pip install flask                           # only additional dep
    PORT=8085 JWT_SECRET=<same as dashboard> python slurm_training_service.py

Then in tt-dashboard (TT_TRAINING_SERVICE_URL or TT_TRAIN_BASE_URL):
    TT_TRAINING_SERVICE_URL=http://<login-node>:8501
    # or TT_TRAIN_BASE_URL=http://<login-node>:8085

JWT_SECRET must match the value used by the dashboard API. The dashboard
sends service-to-service JWTs with svc=="tt-dashboard-api"; this service
validates and accepts those tokens. If JWT_SECRET is omitted, the service
falls back to X-TT-Organization header (dev/testing only).

API contract (mirrors mock-training-service, accepts dashboard format):
    GET  /openapi.yaml                 → OpenAPI 3.0 spec
    GET  /healthz
    GET  /v1/catalog                       → {models, datasets, clusters, trainers, optimizers}
    GET  /v1/jobs                          → {"jobs": [...]}
    POST /v1/jobs                          → Job (accepts dashboard format, maps model→model_config)
    GET  /v1/jobs/{id}                     → Job
    POST /v1/jobs/{id}/cancel              → Job
    GET  /v1/jobs/{id}/metrics             → [MetricPoint, ...]
    GET  /v1/jobs/{id}/logs                → [LogEntry, ...]
    GET  /v1/jobs/{id}/checkpoints         → [Checkpoint, ...]
"""

import hashlib
import hmac
import json
import logging
import os
import re
import struct
import time
import uuid
from base64 import urlsafe_b64decode, urlsafe_b64encode
from datetime import datetime, timezone
from functools import wraps
from typing import Optional


def _rfc3339_utc(dt: datetime | None = None) -> str:
    """Return RFC3339 timestamp with Z suffix (UTC)."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _rfc3339_normalize(s: Optional[str]) -> Optional[str]:
    """Normalize ISO timestamp to RFC3339 with Z suffix."""
    if not s:
        return s
    return s.replace("+00:00", "Z")


from pathlib import Path

from flask import Flask, jsonify, request, g, send_from_directory
from ttml.common.utils import get_tt_metal_runtime_root
from training_types import get_training_type, get_supported_trainers, get_supported_models, TRAINING_TYPES
from job_manager import JobManager, JobStatus, PARTITION_DEVICE_MAPPING

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Observability: use OBS prefix for easy grep/filtering
OBS = logging.getLogger("slurm_training.observability")
OBS.setLevel(logging.DEBUG)
if not OBS.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s OBS %(message)s"))
    OBS.addHandler(h)

app = Flask(__name__)

JWT_SECRET = os.environ.get("JWT_SECRET", "")
PORT = int(os.environ.get("PORT", 8085))


# Prefer /data for jobs when available (compute nodes can write there; /home may not be mounted)
def _default_jobs_base_dir() -> str:
    if os.environ.get("JOBS_BASE_DIR"):
        return os.environ["JOBS_BASE_DIR"]

    # Try /data first (for compute nodes)
    user = os.environ.get("USER", "")
    if user:
        data_jobs = Path(f"/data/{user}/tt-metal/tt-train/sources/examples/gsm8k_finetune/jobs")
        if (Path("/data") / user).exists():
            return str(data_jobs)

    # Fallback to tt_metal_runtime_root relative path
    tt_train_root = f"{get_tt_metal_runtime_root()}/tt-train"
    return f"{tt_train_root}/sources/examples/gsm8k_finetune/jobs"


JOBS_BASE_DIR = _default_jobs_base_dir()

# Default SLURM partition — override with DEFAULT_PARTITION env var.
# Fallback when preferred partition doesn't exist.
DEFAULT_PARTITION = os.environ.get("DEFAULT_PARTITION", "bh_sp5_aisle_c_partial")

# Supported clusters exposed to users. Only these are returned in the catalog.
SUPPORTED_CLUSTERS = {"bh_galaxy", "4bh_glx"}

# Human-readable display names for catalog cluster entries.
CLUSTER_DISPLAY_NAMES = {
    "bh_galaxy": "BH Galaxy",
    "4bh_glx": "4× BH Galaxy",
}

# Cluster size → ordered list of partitions that can satisfy it. First with free nodes wins.
CLUSTER_TO_PARTITIONS = {
    "bh_galaxy": ["bh_sp5_aisle_c_partial"],
    "4bh_glx": ["bh_sp5_aisle_c_partial"],
    # Disabled — not exposed in catalog
    "8xp150": ["bh_lb_single"],
    "4xp150": ["bh_lb_single"],
    "bh_glx": ["bh_single", "bh_sp_5x4x32_C1_C10", "bh_pod_4x32_B45", "bh_pod_4x32_B89"],
    "5x4bh_glx": ["bh_sp_5x4x32_C1_C10"],
}

# Topology per partition: hierarchy and device counts for multi-host clusters.
# mesh_shape is per-node; total_devices = nodes × devices_per_node (devices_per_node = mesh_shape[0]*mesh_shape[1])
PARTITION_TOPOLOGY = {
    "bh_lb_single": {
        "mesh_shape": [8, 1],
        "nodes": 1,
        "pods": 1,
        "nodes_per_pod": 1,
        "total_devices": 8,
        "topology": "1×8×1",
    },
    "bh_lb_multi": {
        "mesh_shape": [8, 1],
        "nodes": 4,
        "pods": 1,
        "nodes_per_pod": 4,
        "total_devices": 32,
        "topology": "4×8×1",
    },
    "bh_single": {
        "mesh_shape": [32, 1],
        "nodes": 1,
        "pods": 1,
        "nodes_per_pod": 1,
        "total_devices": 32,
        "topology": "1×32×1",
    },
    "bh_galaxy": {
        "mesh_shape": [32, 1],
        "nodes": 1,
        "pods": 1,
        "nodes_per_pod": 1,
        "total_devices": 32,
        "topology": "1×32×1",
    },
    "bh_sp5_aisle_c_partial": {
        "mesh_shape": [32, 1],
        "nodes": 4,
        "pods": 1,
        "nodes_per_pod": 4,
        "total_devices": 128,
        "topology": "4×32×1",
    },
    "bh_pod_4x32_B45": {
        "mesh_shape": [32, 1],
        "nodes": 4,
        "pods": 1,
        "nodes_per_pod": 4,
        "total_devices": 128,
        "topology": "4×32×1",
    },
    "bh_pod_4x32_B89": {
        "mesh_shape": [32, 1],
        "nodes": 4,
        "pods": 1,
        "nodes_per_pod": 4,
        "total_devices": 128,
        "topology": "4×32×1",
    },
    "bh_sp_5x4x32_C1_C10": {
        "mesh_shape": [32, 1],
        "nodes": 20,
        "pods": 5,
        "nodes_per_pod": 4,
        "total_devices": 640,
        "topology": "5×4×32×1",
    },
}

# No fallback partitions — jobs must stay within the allowed partition.
PARTITION_FALLBACK = {}


def _get_cluster_info(cluster: str) -> dict:
    """Return cluster catalog entry with partition and topology."""
    partitions = manager.get_available_partitions()
    p = _resolve_partition(cluster, partitions=partitions)
    topo = PARTITION_TOPOLOGY.get(p)
    if not topo:
        is_lb = p and "lb" in p.lower()
        topo = {
            "mesh_shape": [8, 1] if is_lb else [32, 1],
            "nodes": 1,
            "pods": 1,
            "nodes_per_pod": 1,
            "total_devices": 8 if is_lb else 32,
            "topology": "1×8×1" if is_lb else "1×32×1",
        }
    return {
        "id": cluster,
        "display_name": CLUSTER_DISPLAY_NAMES.get(cluster, cluster),
        "partition": p,
        "mesh_shape": topo["mesh_shape"],
        "topology": topo,
    }


def _resolve_partition(cluster: str, partitions: list = None) -> str:
    """Resolve cluster to an available SLURM partition.
    Uses CLUSTER_TO_PARTITIONS to try partitions in order; prefers those with
    free nodes (idle/mix) so jobs schedule immediately when possible.
    """
    if partitions is None:
        partitions = manager.get_available_partitions()
    part_by_name = {p["name"]: p for p in partitions}
    candidates = CLUSTER_TO_PARTITIONS.get(cluster)

    if candidates:
        # Prefer partition with free nodes (can schedule now)
        for name in candidates:
            p = part_by_name.get(name)
            if p and p.get("available", True) and p.get("has_free_nodes"):
                return name
        # Else first partition that exists (job will queue)
        for name in candidates:
            p = part_by_name.get(name)
            if p and p.get("available", True):
                return name
        # Fallback: first that exists even if not "available"
        for name in candidates:
            if name in part_by_name:
                return name

    # Unknown cluster: try prefix match (legacy)
    for p in partitions:
        if cluster.lower() in p["name"].lower() and p.get("has_free_nodes"):
            return p["name"]
    for p in partitions:
        if cluster.lower() in p["name"].lower():
            return p["name"]

    # Last resort: first available partition
    for p in partitions:
        if p.get("available", True) and p.get("has_free_nodes"):
            return p["name"]
    for p in partitions:
        if p.get("available", True):
            return p["name"]
    for p in partitions:
        return p["name"]
    return DEFAULT_PARTITION


manager = JobManager(jobs_base_dir=JOBS_BASE_DIR)


# ---------------------------------------------------------------------------
# Observability: request logging, job queue snapshots, status transitions
# ---------------------------------------------------------------------------


def _truncate(s: str, max_len: int = 500) -> str:
    """Truncate string for safe logging."""
    s = str(s)
    return s[:max_len] + "..." if len(s) > max_len else s


def _obs_tag(tag: str, width: int = 14) -> str:
    """Return aligned log tag for consistent column alignment."""
    return (f"[{tag}]").ljust(width)


def _sanitize_body(body) -> str:
    """Safe representation of request body for logging."""
    if body is None:
        return "<empty>"
    try:
        s = json.dumps(body) if isinstance(body, dict) else str(body)
        return _truncate(s, 800)
    except Exception:
        return "<non-serializable>"


@app.before_request
def _obs_log_request():
    g.obs_request_id = str(uuid.uuid4())[:8]
    g.obs_start = time.perf_counter()
    body = None
    if request.method in ("POST", "PUT", "PATCH") and request.is_json:
        try:
            body = request.get_json(silent=True)
        except Exception:
            body = "<parse_error>"
    OBS.info(
        "%s request_id=%s method=%s path=%s query=%s body=%s",
        _obs_tag("REQ"),
        g.obs_request_id,
        request.method,
        request.path,
        dict(request.args) if request.args else {},
        _sanitize_body(body),
    )


@app.after_request
def _obs_log_response(response):
    req_id = getattr(g, "obs_request_id", "?")
    elapsed_ms = (time.perf_counter() - getattr(g, "obs_start", time.perf_counter())) * 1000
    OBS.info(
        "%s request_id=%s status=%s elapsed_ms=%.1f",
        _obs_tag("RESP"),
        req_id,
        response.status_code,
        elapsed_ms,
    )
    return response


def _obs_log_job_queue(reason: str):
    """Log full job queue snapshot (all jobs, their statuses, SLURM IDs)."""
    snapshot = []
    for dash_id, meta in _state.items():
        slurm_id = meta.get("slurm_job_id")
        status = meta.get("status", "?")
        model = meta.get("model", "")
        created = meta.get("created_at", "")[:19] if meta.get("created_at") else ""
        snapshot.append(
            {
                "dash_id": dash_id[:8],
                "slurm_id": slurm_id,
                "status": status,
                "model": model[:30] if model else "",
                "created": created,
            }
        )
    OBS.info(
        "%s reason=%s count=%d jobs=%s",
        _obs_tag("QUEUE"),
        reason,
        len(snapshot),
        snapshot,
    )


def _obs_log_status_transition(dash_id: str, old_status: str, new_status: str, meta: dict):
    """Log when a job's status changes."""
    slurm_id = meta.get("slurm_job_id")
    model = meta.get("model", "")
    OBS.info(
        "%s dash_id=%s slurm_id=%s model=%s transition=%s -> %s",
        _obs_tag("STATUS"),
        dash_id[:8],
        slurm_id,
        model[:30] if model else "",
        old_status,
        new_status,
    )


# ---------------------------------------------------------------------------
# Minimal JWT validation (HMAC-SHA256, matches the Go jwt package)
# ---------------------------------------------------------------------------


def _b64_decode(s: str) -> bytes:
    # Add padding if needed
    s += "=" * (-len(s) % 4)
    return urlsafe_b64decode(s)


def _validate_jwt(token: str) -> Optional[dict]:
    """Validate a HS256 JWT signed with JWT_SECRET. Returns claims or None."""
    if not JWT_SECRET:
        return None
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b64, payload_b64, sig_b64 = parts
        signing_input = f"{header_b64}.{payload_b64}".encode()
        expected_sig = hmac.new(JWT_SECRET.encode(), signing_input, hashlib.sha256).digest()
        actual_sig = _b64_decode(sig_b64)
        if not hmac.compare_digest(expected_sig, actual_sig):
            return None
        claims = json.loads(_b64_decode(payload_b64))
        # Check expiry
        if "exp" in claims and claims["exp"] < time.time():
            return None
        return claims
    except Exception:
        return None


def require_auth(f):
    """Decorator that extracts org_id from JWT or X-TT-Organization header."""

    @wraps(f)
    def decorated(*args, **kwargs):
        if JWT_SECRET:
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer "):
                OBS.info(
                    "%s rejected path=%s reason=no_bearer_token",
                    _obs_tag("AUTH"),
                    request.path,
                )
                return (
                    jsonify({"error": {"message": "Authorization: Bearer <token> required"}}),
                    401,
                )
            claims = _validate_jwt(auth[7:])
            if claims is None:
                OBS.info(
                    "%s rejected path=%s reason=invalid_or_expired_token",
                    _obs_tag("AUTH"),
                    request.path,
                )
                return jsonify({"error": {"message": "Invalid or expired token"}}), 401
            # Dashboard sends svc == "tt-dashboard-api"; some implementations use service_id
            allowed_svc = claims.get("svc") == "tt-dashboard-api" or claims.get("service_id") == "tt-dashboard-api"
            if not allowed_svc:
                OBS.info(
                    "%s rejected path=%s reason=wrong_service claims_svc=%s",
                    _obs_tag("AUTH"),
                    request.path,
                    claims.get("svc") or claims.get("service_id"),
                )
                return (
                    jsonify({"error": {"message": "Token not issued for this service"}}),
                    403,
                )
            g.org_id = claims.get("org_id", "")
        else:
            org_id = request.headers.get("X-TT-Organization", "")
            if not org_id:
                OBS.info(
                    "%s rejected path=%s reason=no_org_header",
                    _obs_tag("AUTH"),
                    request.path,
                )
                return (
                    jsonify({"error": {"message": "JWT_SECRET not set and X-TT-Organization header missing"}}),
                    401,
                )
            g.org_id = org_id
        return f(*args, **kwargs)

    return decorated


# ---------------------------------------------------------------------------
# Job state persistence (augments JobManager with org_id + dashboard fields)
# ---------------------------------------------------------------------------

STATE_FILE = Path(JOBS_BASE_DIR) / "dashboard_jobs.json"


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


# dashboard_id → {slurm_job_id, org_id, created_by_user_id, model, dataset_url,
#                  optimizer, trainer, cluster_size, training_params, created_at}
_state: dict = _load_state()


def _slurm_status_to_dashboard(status: str) -> str:
    mapping = {
        JobStatus.PENDING.value: "queued",
        JobStatus.RUNNING.value: "running",
        JobStatus.COMPLETED.value: "completed",
        JobStatus.FAILED.value: "failed",
        JobStatus.CANCELLED.value: "cancelled",
        JobStatus.UNKNOWN.value: "queued",
    }
    return mapping.get(status, "queued")


def _build_job_response(dash_id: str, meta: dict) -> dict:
    """Construct the Job JSON response from stored metadata + live SLURM status."""
    slurm_id = meta.get("slurm_job_id")
    slurm_info = manager.get_job(slurm_id) if slurm_id else None

    # Always query live SLURM status so we detect externally-cancelled jobs
    # (scancel from CLI, dashboard, etc.) immediately
    live_status_str = None
    if slurm_id:
        live_status = manager.get_job_status(slurm_id)
        if live_status != JobStatus.UNKNOWN:
            live_status_str = _slurm_status_to_dashboard(live_status.value)
            old_status = meta.get("last_known_status") or meta.get("status", "queued")
            if live_status_str != old_status:
                meta["last_known_status"] = live_status_str
                _save_state(_state)

    old_status = meta.get("last_known_status") or meta.get("status", "queued")

    if slurm_info:
        # Prefer live SLURM status over cache (catches external cancels).
        # When sacct is disabled, live_status is UNKNOWN for finished jobs - trust stored
        # terminal state (cancelled/completed/failed) from manual updates or dashboard cancel.
        if live_status_str:
            status = live_status_str
        elif meta.get("last_known_status") in ("cancelled", "completed", "failed"):
            status = meta["last_known_status"]
        elif meta.get("status") in ("cancelled", "completed", "failed"):
            status = meta["status"]
        else:
            status = _slurm_status_to_dashboard(slurm_info.status)
        started_at = slurm_info.start_time
        completed_at = (
            slurm_info.end_time
            if slurm_info.status
            in (
                JobStatus.COMPLETED.value,
                JobStatus.FAILED.value,
                JobStatus.CANCELLED.value,
            )
            else None
        )
        output_dir = slurm_info.output_dir
        # Observability: log status transitions and persist last_known_status
        if status != old_status:
            _obs_log_status_transition(dash_id, old_status, status, meta)
            meta["last_known_status"] = status
            _save_state(_state)
    else:
        # When no slurm_info: prefer live status, then last_known_status (often more
        # accurate than status for completed/cancelled jobs), then status
        status = live_status_str if live_status_str else meta.get("last_known_status") or meta.get("status", "queued")
        started_at = None
        completed_at = None
        output_dir = meta.get("output_dir", "")

    return {
        "id": dash_id,
        "org_id": meta.get("org_id", ""),
        "created_by_user_id": meta.get("created_by_user_id", ""),
        "status": status,
        "trainer": meta.get("trainer", "sft"),
        "model": meta.get("model", ""),
        "dataset_url": meta.get("dataset_url", ""),
        "optimizer": meta.get("optimizer", "adamw"),
        "training_params": meta.get("training_params", {}),
        "cluster": meta.get("cluster") or meta.get("compute_size") or meta.get("cluster_size", ""),
        "estimated_cost_cents": meta.get("estimated_cost_cents"),
        "actual_cost_cents": None,
        "error_message": None,
        "started_at": _rfc3339_normalize(started_at) if started_at else None,
        "completed_at": _rfc3339_normalize(completed_at) if completed_at else None,
        "created_at": _rfc3339_normalize(meta.get("created_at", "")) or "",
        "updated_at": _rfc3339_normalize(meta.get("created_at", "")) or "",
        "_output_dir": output_dir,  # internal, stripped before sending
    }


# ---------------------------------------------------------------------------
# Metric / log parsing
# ---------------------------------------------------------------------------

# output.txt: val_loss only when validation ran (every eval_every steps)
# With val: "LR: 0.0001, training_loss: 0.97, val_loss: 1.01, step: 20, epoch: 1"
# Without:   "LR: 0.0001, training_loss: 0.97, step: 21, epoch: 1"
_METRIC_RE = re.compile(
    r"LR:\s*(?P<lr>[\d.e+-]+),\s*"
    r"training_loss:\s*(?P<train_loss>[\d.e+-]+),\s*"
    r"(?:val_loss:\s*(?P<val_loss>[\d.e+-]+),\s*)?"
    r"step:\s*(?P<step>\d+),\s*"
    r"epoch:\s*(?P<epoch>\d+)"
)


def _parse_metrics(output_dir: str) -> list:
    path = Path(output_dir) / "output.txt"
    if not path.exists():
        return []
    points = []
    for line in path.read_text().splitlines():
        m = _METRIC_RE.search(line)
        if not m:
            continue
        step = int(m.group("step"))
        train_loss = float(m.group("train_loss"))
        val_loss_grp = m.group("val_loss")
        lr = float(m.group("lr"))
        epoch = int(m.group("epoch"))
        pt = {
            "step": step,
            "epoch": epoch,
            "train_loss": train_loss,
            "grad_norm": 0.0,  # not emitted by the training script
            "learning_rate": lr,
        }
        if val_loss_grp:
            val_loss = float(val_loss_grp)
            if val_loss > 0:
                pt["val_loss"] = val_loss
        points.append(pt)
    return points


def _parse_logs(output_dir: str) -> list:
    entries = []
    idx = 0

    def _add(ts: str, typ: str, msg: str, step=None):
        nonlocal idx
        entry = {
            "id": f"log_{idx}",
            "timestamp": ts,
            "type": typ,
            "message": msg,
        }
        if step is not None:
            entry["step"] = step
        entries.append(entry)
        idx += 1

    now_iso = _rfc3339_utc()

    # Prefer training_stdout/stderr.log; fallback to SLURM slurm_*.out / slurm_*.err
    log_sources = [
        ("training_stdout.log", "stdout", "info"),
        ("training_stderr.log", "stderr", "error"),
    ]
    output_path = Path(output_dir)
    for logfile, source, typ in log_sources:
        path = output_path / logfile
        if not path.exists():
            # Fallback: SLURM writes to slurm_<jobid>.out / slurm_<jobid>.err
            candidates = sorted(output_path.glob("slurm_*.out" if "out" in logfile else "slurm_*.err"))
            path = candidates[-1] if candidates else None
        if path and path.exists():
            for line in path.read_text(errors="replace").splitlines()[-200:]:
                line = line.strip()
                if not line:
                    continue
                _add(now_iso, typ, f"[{source}] {line}")

    # Surface checkpoint lines from output.txt (only when val_loss was computed)
    out_path = Path(output_dir) / "output.txt"
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            m = _METRIC_RE.search(line)
            if m and m.group("val_loss"):
                step = int(m.group("step"))
                tl = float(m.group("train_loss"))
                vl = float(m.group("val_loss"))
                _add(
                    now_iso,
                    "checkpoint",
                    f"step {step} | train_loss: {tl:.4f} | val_loss: {vl:.4f}",
                    step=step,
                )

    return entries


def _parse_checkpoints(output_dir: str) -> list:
    """
    Treat each metric line that has a non-zero val_loss as a checkpoint boundary.
    Real checkpoint files could also be scanned here if the training script saves them.
    """
    metrics = _parse_metrics(output_dir)
    checkpoints = []
    for pt in metrics:
        if "val_loss" not in pt:
            continue
        checkpoints.append(
            {
                "id": f"ckpt_{pt['step']}",
                "step": pt["step"],
                "epoch": pt["epoch"],
                "metrics": {"train_loss": pt["train_loss"], "val_loss": pt["val_loss"]},
                "created_at": _rfc3339_utc(),
                "ckpt_type": "full",
            }
        )
    return checkpoints


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/openapi.yaml")
def openapi_spec():
    """Serve OpenAPI 3.0 specification."""
    spec_dir = Path(__file__).parent
    return send_from_directory(spec_dir, "openapi.yaml", mimetype="application/x-yaml")


@app.get("/healthz")
def healthz():
    slurm_ok = _slurm_available()
    OBS.info("%s slurm=%s", _obs_tag("HEALTHZ"), slurm_ok)
    return jsonify({"status": "ok", "slurm": slurm_ok})


def _slurm_available() -> bool:
    import shutil

    return shutil.which("squeue") is not None


@app.get("/v1/jobs")
@require_auth
def list_jobs():
    org_id = g.org_id
    jobs = []
    for dash_id, meta in _state.items():
        if meta.get("org_id") != org_id:
            continue
        resp = _build_job_response(dash_id, meta)
        resp.pop("_output_dir", None)
        jobs.append(resp)
    _obs_log_job_queue("list_jobs")
    OBS.info("%s org_id=%s count=%d", _obs_tag("LIST"), _truncate(org_id, 12), len(jobs))
    return jsonify({"jobs": jobs})


# Capabilities: what the server/demo actually supports (dashboard should restrict or warn)
SUPPORTED_TRAINERS = get_supported_trainers()
SUPPORTED_OPTIMIZERS = {"adamw"}
SUPPORTED_DATASETS = {"gsm8k"}  # others work in gsm8k_finetune but less tested


@app.get("/v1/catalog")
def catalog():
    """Return valid model, dataset, cluster values and supported flags for UI (no auth).
    Dashboard should use `supported` to enable/disable options or show 'coming soon'.
    """
    log.info("=== CATALOG REQUEST ===")
    log.info("Fetching available models, datasets, trainers, optimizers, and clusters")

    # Generate models dynamically from training registry
    all_models = get_supported_models()
    model_display_names = {
        "tinyllama": "TinyLlama 1.1B",
        "gpt2": "GPT-2",
        "llama8b": "Llama 3.1 8B",
    }

    # Get model configs from registry (they already have proper TT_METAL_RUNTIME_ROOT paths)
    model_configs = {}
    for training_config in TRAINING_TYPES.values():
        model_configs.update(training_config.model_configs)

    models = []
    for model_id in sorted(all_models):
        # Convert full path back to relative path for API response
        full_config_path = model_configs.get(model_id, "")
        relative_config_path = (
            full_config_path.split("/configs/")[-1]
            if "/configs/" in full_config_path
            else f"model_configs/{model_id}.yaml"
        )

        # Mark smaller models as supported by default, larger ones may need more resources
        is_supported = model_id in {"tinyllama", "gpt2"}

        models.append(
            {
                "id": model_id,
                "display_name": model_display_names.get(model_id, model_id.title()),
                "model_config": f"configs/{relative_config_path}",
                "supported": is_supported,
            }
        )

    # Generate trainers dynamically from training registry
    trainer_display_names = {
        "sft": "SFT",
        "lora": "LoRA",
        "grpo": "GRPO",
    }

    trainers = []
    for trainer_id in sorted(TRAINING_TYPES.keys()):
        trainers.append(
            {
                "id": trainer_id,
                "display_name": trainer_display_names.get(trainer_id, trainer_id.upper()),
                "supported": True,  # All registered trainers are supported
            }
        )

    # Add known future trainers that aren't implemented yet
    future_trainers = {"grpo"}
    for trainer_id in sorted(future_trainers - set(TRAINING_TYPES.keys())):
        trainers.append(
            {
                "id": trainer_id,
                "display_name": trainer_display_names.get(trainer_id, trainer_id.upper()),
                "supported": False,
            }
        )

    catalog_data = {
        "supported": {
            "trainers": list(SUPPORTED_TRAINERS),
            "optimizers": list(SUPPORTED_OPTIMIZERS),
        },
        "models": models,
        "datasets": [
            {"id": "gsm8k", "display_name": "GSM8K", "supported": True},
            {"id": "math_qa", "display_name": "Math QA", "supported": True},
            {"id": "aqua_rat", "display_name": "AQuA-RAT", "supported": True},
            {"id": "svamp", "display_name": "SVAMP", "supported": True},
            {"id": "mawps", "display_name": "MAWPS", "supported": True},
        ],
        "trainers": trainers,
        "optimizers": [
            {"id": "adamw", "display_name": "AdamW", "supported": True},
            {"id": "sgd", "display_name": "SGD", "supported": False},
            {"id": "muon", "display_name": "Muon", "supported": False},
        ],
        "clusters": [_get_cluster_info(c) for c in SUPPORTED_CLUSTERS],
    }

    log.info("=== CATALOG RESPONSE ===")
    log.info("Available models: %s", [m["id"] for m in catalog_data["models"]])
    log.info("Supported trainers: %s", [t["id"] for t in catalog_data["trainers"] if t["supported"]])
    log.info("Supported optimizers: %s", [o["id"] for o in catalog_data["optimizers"] if o["supported"]])
    log.info("Available clusters: %s", [c["id"] for c in catalog_data["clusters"]])
    log.info("Full catalog: %s", json.dumps(catalog_data, indent=2))

    return jsonify(catalog_data)


@app.post("/v1/jobs")
@require_auth
def create_job():
    org_id = g.org_id
    body = request.get_json(force=True)

    # Log the raw request from frontend
    log.info("=== RECEIVED TRAINING JOB REQUEST ===")
    log.info("Request body: %s", json.dumps(body, indent=2))
    log.info("Organization ID: %s", org_id)

    model = body.get("model", "")
    dataset_url = body.get("dataset_url", "")
    cluster = body.get("cluster") or body.get("compute_size") or body.get("cluster_size", "")
    if not model or not dataset_url or not cluster:
        OBS.info("%s validation_failed missing=model|dataset_url|cluster", _obs_tag("CREATE"))
        return (
            jsonify({"error": {"message": "model, dataset_url, and cluster are required"}}),
            400,
        )
    if cluster not in SUPPORTED_CLUSTERS:
        OBS.info(
            "%s unsupported_cluster cluster=%s supported=%s", _obs_tag("CREATE"), cluster, sorted(SUPPORTED_CLUSTERS)
        )
        return (
            jsonify(
                {
                    "error": {
                        "message": f"Cluster '{cluster}' is not available. Supported: {sorted(SUPPORTED_CLUSTERS)}",
                        "code": "unsupported_cluster",
                    }
                }
            ),
            400,
        )

    raw_params = body.get("training_params") or {}
    if not isinstance(raw_params, dict) or "type" not in raw_params or "params" not in raw_params:
        return (
            jsonify(
                {
                    "error": {
                        "message": f"training_params must be {{ type: {('|'.join(sorted(get_supported_trainers())))}, params: {{...}} }}",
                        "code": "invalid_training_params",
                    }
                }
            ),
            400,
        )
    trainer = raw_params["type"]

    raw_optimizer = body.get("optimizer_params")
    if not isinstance(raw_optimizer, dict) or "type" not in raw_optimizer or "params" not in raw_optimizer:
        return (
            jsonify(
                {
                    "error": {
                        "message": "optimizer_params must be { type: adamw|sgd|muon, params: {...} }",
                        "code": "invalid_optimizer_params",
                    }
                }
            ),
            400,
        )
    optimizer = raw_optimizer["type"]
    optimizer_params = dict(raw_optimizer.get("params", {}))

    supported_trainers = get_supported_trainers()
    if trainer not in supported_trainers:
        OBS.info(
            "%s unsupported_trainer trainer=%s supported=%s",
            _obs_tag("CREATE"),
            trainer,
            supported_trainers,
        )
        return (
            jsonify(
                {
                    "error": {
                        "message": f"Unsupported trainer: '{trainer}'. Supported: {sorted(supported_trainers)}. Check GET /v1/catalog for capabilities.",
                        "code": "unsupported_trainer",
                    }
                }
            ),
            400,
        )
    if optimizer not in SUPPORTED_OPTIMIZERS:
        OBS.info(
            "%s unsupported_optimizer optimizer=%s supported=%s",
            _obs_tag("CREATE"),
            optimizer,
            SUPPORTED_OPTIMIZERS,
        )
        return (
            jsonify(
                {
                    "error": {
                        "message": f"Unsupported optimizer: '{optimizer}'. Supported: {sorted(SUPPORTED_OPTIMIZERS)}. Check GET /v1/catalog for capabilities.",
                        "code": "unsupported_optimizer",
                    }
                }
            ),
            400,
        )

    training_params = dict(raw_params["params"])

    # Merge optimizer params into training_params for downstream
    training_params.update(optimizer_params)

    # Get training type configuration and validate parameters
    training_config = get_training_type(trainer)
    try:
        training_params = training_config.param_validator(training_params)
    except ValueError as e:
        return (
            jsonify(
                {
                    "error": {
                        "message": f"Parameter validation failed: {str(e)}",
                        "code": "invalid_training_params",
                    }
                }
            ),
            400,
        )

    # Map model ID → model_config (dashboard format)
    model_config = training_config.model_configs.get(model.strip().lower())
    if model_config:
        training_params["model_config"] = model_config
    if "max_steps" not in training_params and "epochs" in training_params:
        training_params["max_steps"] = training_params["epochs"] * 20

    # Log parsed parameters
    log.info("=== PARSED PARAMETERS ===")
    log.info("Model: %s -> config: %s", model, model_config)
    log.info("Dataset URL: %s", dataset_url)
    log.info("Trainer: %s", trainer)
    log.info("Optimizer: %s", optimizer)
    log.info("Cluster requested: %s", cluster)
    log.info("Training params: %s", json.dumps(training_params, indent=2))

    partition = _resolve_partition(cluster)
    # Derive node count from cluster: 4bh_glx needs 4 nodes, bh_galaxy needs 1
    cluster_nodes = 4 if cluster == "4bh_glx" else 1
    log.info("=== CLUSTER RESOLUTION ===")
    log.info("Cluster '%s' resolved to partition: %s (nodes=%d)", cluster, partition, cluster_nodes)
    batch_size = training_params.get("batch_size", 8)
    if "lb" not in partition.lower() and batch_size % 32 != 0:
        return (
            jsonify(
                {
                    "error": {
                        "message": f"Galaxy (mesh [32,1]) requires batch_size divisible by 32; got {batch_size}",
                        "code": "invalid_batch_size",
                    }
                }
            ),
            400,
        )

    OBS.info(
        "%s org_id=%s model=%s cluster=%s partition=%s trainer=%s",
        _obs_tag("CREATE"),
        _truncate(g.org_id, 12),
        model[:40],
        cluster,
        partition,
        trainer,
    )

    # Build SLURM config from dashboard training params (aligned with streamlit_finetune_app.py)
    sched = training_params.get("scheduler_config") or {}
    slurm_config = {
        "batch_size": training_params.get("batch_size", 8),
        "max_steps": training_params.get("max_steps") or training_params.get("epochs", 3) * 20,
        "max_lr": sched.get("max_lr", 1e-4),
        "min_lr": sched.get("min_lr", 3e-5),
        "eval_every": training_params.get("eval_every", 20),
        "gradient_accumulation": training_params.get("gradient_accumulation_steps", 1),
        "warmup_steps": sched.get("warmup_steps", 20),
        "hold_steps": sched.get("hold_steps", 40),
        "validation_batch_size": training_params.get("validation_batch_size", 4),
        "max_seq_length": training_params.get("max_sequence_length", 512),
        "enable_ddp": True,  # Always enabled; dashboard need not send
        # mesh_shape derived by job_manager from partition (lb→[8,1], galaxy→[32,1])
    }
    if training_params.get("model_config"):
        slurm_config["model_config"] = training_params["model_config"]
    # Pass dataset for training script (HF name or s3:// URL)
    slurm_config["dataset"] = dataset_url
    if not slurm_config["dataset"]:
        return (
            jsonify(
                {
                    "error": {
                        "message": "dataset_url is required",
                        "code": "missing_dataset",
                    }
                }
            ),
            400,
        )

    job_name = f"tt_{model.replace('-', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Log the final SLURM configuration that would be submitted
    log.info("=== SLURM JOB CONFIGURATION ===")
    log.info("Job name: %s", job_name)
    log.info("Partition: %s", partition)
    log.info("Nodes: %d", cluster_nodes)
    log.info("SLURM config: %s", json.dumps(slurm_config, indent=2))
    OBS.info("%s slurm_config=%s job_name=%s", _obs_tag("CREATE"), slurm_config, job_name)

    log.info("=== SUBMITTING TO SLURM ===")
    log.info("Attempting sbatch submission with:")
    log.info("  - Config: %s", slurm_config)
    log.info("  - Partition: %s", partition)
    log.info("  - Nodes: %d", cluster_nodes)
    log.info("  - Job name: %s", job_name)

    success, msg, slurm_info = manager.submit_job(
        training_config=training_config,
        config=slurm_config,
        partition=partition,
        nodes=cluster_nodes,
        job_name=job_name,
    )

    log.info("=== SLURM SUBMISSION RESULT ===")
    log.info("Success: %s", success)
    log.info("Message: %s", msg)
    if slurm_info:
        log.info("SLURM job ID: %s", slurm_info.job_id)
        log.info("Output directory: %s", slurm_info.output_dir)
    else:
        log.info("No SLURM info returned (submission likely failed)")

    # If primary partition failed (node config not available, etc.), try fallback
    if not success and partition != DEFAULT_PARTITION:
        fallback = PARTITION_FALLBACK.get(partition)
        if not fallback and "lb" not in partition.lower():
            fallback = DEFAULT_PARTITION  # try LoudBox as universal fallback
        if fallback and fallback != partition:
            log.info("=== TRYING FALLBACK PARTITION ===")
            log.info("Primary partition '%s' failed, trying fallback: %s", partition, fallback)
            OBS.info(
                "%s retry_fallback partition=%s fallback=%s",
                _obs_tag("CREATE"),
                partition,
                fallback,
            )
            success, msg, slurm_info = manager.submit_job(
                training_config=training_config,
                config=slurm_config,
                partition=fallback,
                nodes=cluster_nodes,
                job_name=job_name,
            )
            log.info("=== FALLBACK SUBMISSION RESULT ===")
            log.info("Success: %s", success)
            log.info("Message: %s", msg)
            if success:
                partition = fallback
                log.info("Using fallback partition: %s", partition)
            else:
                log.info("Fallback partition also failed")

    dash_id = str(uuid.uuid4())
    now = _rfc3339_utc()

    log.info("=== CREATING DASHBOARD JOB RECORD ===")
    log.info("Dashboard job ID: %s", dash_id)
    log.info("Timestamp: %s", now)

    if not success:
        OBS.info(
            "%s submit_failed dash_id=%s error=%s",
            _obs_tag("CREATE"),
            dash_id[:8],
            _truncate(msg, 200),
        )
        log.warning("sbatch failed: %s — storing job as failed", msg)
        meta = {
            "slurm_job_id": None,
            "org_id": org_id,
            "created_by_user_id": str(uuid.uuid4()),
            "model": model,
            "dataset_url": dataset_url,
            "optimizer": optimizer,
            "trainer": trainer,
            "cluster": cluster,
            "training_params": training_params,
            "estimated_cost_cents": None,
            "status": "failed",
            "created_at": now,
            "output_dir": "",
            "slurm_error": msg,
            "last_known_status": "failed",
        }
        _state[dash_id] = meta
        _save_state(_state)
        _obs_log_job_queue("create_job_failed")
    else:
        meta = {
            "slurm_job_id": slurm_info.job_id,
            "org_id": org_id,
            "created_by_user_id": str(uuid.uuid4()),
            "model": model,
            "dataset_url": dataset_url,
            "optimizer": optimizer,
            "trainer": trainer,
            "cluster": cluster,
            "training_params": training_params,
            "estimated_cost_cents": None,
            "status": "queued",
            "last_known_status": "queued",
            "created_at": now,
            "output_dir": slurm_info.output_dir,
        }
        OBS.info(
            "%s submitted dash_id=%s slurm_id=%s output_dir=%s",
            _obs_tag("CREATE"),
            dash_id[:8],
            slurm_info.job_id,
            slurm_info.output_dir,
        )
        log.info("Submitted SLURM job %s for dashboard job %s", slurm_info.job_id, dash_id)
        _state[dash_id] = meta
        _save_state(_state)
        _obs_log_job_queue("create_job")

    resp = _build_job_response(dash_id, meta)
    resp.pop("_output_dir", None)

    log.info("=== FINAL RESPONSE TO DASHBOARD ===")
    log.info("HTTP Status: 201")
    log.info("Response body: %s", json.dumps(resp, indent=2))
    log.info("==========================================")

    return jsonify(resp), 201


@app.get("/v1/jobs/<dash_id>")
@require_auth
def get_job(dash_id):
    meta = _state.get(dash_id)
    if not meta or meta.get("org_id") != g.org_id:
        OBS.info("%s not_found dash_id=%s", _obs_tag("GET"), dash_id[:8])
        return jsonify({"error": {"message": "job not found"}}), 404
    resp = _build_job_response(dash_id, meta)
    resp.pop("_output_dir", None)
    OBS.info("%s dash_id=%s status=%s", _obs_tag("GET"), dash_id[:8], resp.get("status"))
    return jsonify(resp)


@app.post("/v1/jobs/<dash_id>/cancel")
@require_auth
def cancel_job(dash_id):
    meta = _state.get(dash_id)
    if not meta or meta.get("org_id") != g.org_id:
        OBS.info("%s not_found dash_id=%s", _obs_tag("CANCEL"), dash_id[:8])
        return jsonify({"error": {"message": "job not found"}}), 404

    slurm_id = meta.get("slurm_job_id")
    prev_status = meta.get("last_known_status") or meta.get("status", "?")
    OBS.info(
        "%s dash_id=%s slurm_id=%s prev_status=%s",
        _obs_tag("CANCEL"),
        dash_id[:8],
        slurm_id,
        prev_status,
    )

    if slurm_id:
        success, msg = manager.cancel_job(slurm_id)
        OBS.info(
            "%s slurm_cancel slurm_id=%s success=%s msg=%s",
            _obs_tag("CANCEL"),
            slurm_id,
            success,
            _truncate(msg, 100),
        )
        if not success:
            log.warning("cancel failed for slurm job %s: %s", slurm_id, msg)
            return (
                jsonify({"error": {"message": f"Failed to cancel SLURM job: {msg}"}}),
                500,
            )

    meta["status"] = "cancelled"
    meta["last_known_status"] = "cancelled"
    _save_state(_state)
    _obs_log_job_queue("cancel_job")

    resp = _build_job_response(dash_id, meta)
    resp.pop("_output_dir", None)
    return jsonify(resp)


@app.get("/v1/jobs/<dash_id>/metrics")
@require_auth
def get_metrics(dash_id):
    meta = _state.get(dash_id)
    if not meta or meta.get("org_id") != g.org_id:
        return jsonify({"error": {"message": "job not found"}}), 404

    output_dir = _resolve_output_dir(meta)
    metrics = _parse_metrics(output_dir) if output_dir else []
    OBS.info("%s dash_id=%s points=%d", _obs_tag("METRICS"), dash_id[:8], len(metrics))
    return jsonify(metrics)


@app.get("/v1/jobs/<dash_id>/logs")
@require_auth
def get_logs(dash_id):
    meta = _state.get(dash_id)
    if not meta or meta.get("org_id") != g.org_id:
        return jsonify({"error": {"message": "job not found"}}), 404

    output_dir = _resolve_output_dir(meta)
    logs = _parse_logs(output_dir) if output_dir else []
    OBS.info("%s dash_id=%s entries=%d", _obs_tag("LOGS"), dash_id[:8], len(logs))
    return jsonify(logs)


@app.get("/v1/jobs/<dash_id>/checkpoints")
@require_auth
def get_checkpoints(dash_id):
    meta = _state.get(dash_id)
    if not meta or meta.get("org_id") != g.org_id:
        return jsonify({"error": {"message": "job not found"}}), 404

    output_dir = _resolve_output_dir(meta)
    checkpoints = _parse_checkpoints(output_dir) if output_dir else []
    OBS.info("%s dash_id=%s count=%d", _obs_tag("CHECKPOINTS"), dash_id[:8], len(checkpoints))
    return jsonify(checkpoints)


def _resolve_output_dir(meta: dict) -> Optional[str]:
    """Return the job output directory, checking both stored value and JobManager."""
    output_dir = meta.get("output_dir", "")
    if output_dir and Path(output_dir).exists():
        return output_dir
    slurm_id = meta.get("slurm_job_id")
    if slurm_id:
        info = manager.get_job(slurm_id)
        if info and info.output_dir:
            meta["output_dir"] = info.output_dir  # cache it
            return info.output_dir
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("Observability enabled — filter logs with: grep OBS (request_id, QUEUE, STATUS, CREATE, etc.)")
    if JWT_SECRET:
        log.info("JWT auth enabled")
    else:
        log.warning("JWT_SECRET not set — auth disabled, using X-TT-Organization header")

    log.info(
        "Available SLURM partitions: %s",
        [p["name"] for p in manager.get_available_partitions()],
    )
    log.info("Default partition: %s", DEFAULT_PARTITION)
    log.info("Jobs directory: %s", JOBS_BASE_DIR)
    log.info("Listening on :%d", PORT)
    log.info("Set TT_TRAIN_BASE_URL=http://<this-host>:%d in tt-dashboard", PORT)

    app.run(host="0.0.0.0", port=PORT, debug=False)
