"""LoopContext — the single seam every Agent Loop handler reads/writes through.

Bundles the already-built run artifacts (Checkpoint, Ledger, Manifest, Run) and
the playbook index, plus the small helpers handlers need (telemetry, events,
baseline profile). A handler never opens a file directly — it goes through ctx,
so the inter-stage contract is `ctx.state[...]` with the schema fixed in the PLAN.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import router
from .checkpoint import Checkpoint
from .clock import utc_ts
from .events import write_event
from .ledger import Ledger
from .run import Run


class LoopContext:
    def __init__(self, run: Run, state: dict[str, Any], manifest: dict[str, Any], index: list[dict[str, Any]]):
        self.run = run
        self.state = state
        self.manifest = manifest
        self.index = index
        self.checkpoint = Checkpoint(run.state_path)
        self.ledger = Ledger(run.ledger_path)
        self._agent_calls = run.dir / "agent_calls.jsonl"
        self._events = run.dir / "events.jsonl"
        self.deps: dict = {}  # injected callables (edit_runner, select_runner) — real by default

    # --- construction --------------------------------------------------------
    @classmethod
    def from_run(
        cls,
        run: Run,
        playbook_dir: str | Path = router.GUIDELINES_DIR,
        index: list[dict[str, Any]] | None = None,
    ) -> "LoopContext":
        state = Checkpoint(run.state_path).load()
        manifest = run.manifest.read()
        if index is None:
            index = router.build_index(playbook_dir)
        return cls(run, state, manifest, index)

    @classmethod
    def from_latest(
        cls,
        runs_root: str | Path = "runs",
        playbook_dir: str | Path = router.GUIDELINES_DIR,
        index: list[dict[str, Any]] | None = None,
    ) -> "LoopContext":
        run = Run.latest(runs_root)
        if run is None:
            raise FileNotFoundError(f"no runs/latest under {runs_root!r} — run the Before Loop first")
        return cls.from_run(run, playbook_dir, index)

    # --- persistence ---------------------------------------------------------
    def save(self) -> None:
        """Atomic checkpoint of the live state (the engine calls this each step)."""
        self.checkpoint.save(self.state)

    # --- reads ---------------------------------------------------------------
    def baseline_profile(self) -> dict[str, Any]:
        """The immutable stage-1 baseline — the fixed reference, NOT what ROUTE reads."""
        return json.loads((self.run.profiles_dir / "baseline_profile.json").read_text())

    def current_profile(self) -> dict[str, Any]:
        """Profile of the latest COMMITTED model — what ROUTE routes on.

        Falls back to the baseline on iteration 0 (nothing committed yet).
        `state['current_profile']` is a run-relative path set by COMMIT.
        """
        rel = self.state.get("current_profile")
        path = (self.run.dir / rel) if rel else (self.run.profiles_dir / "baseline_profile.json")
        return json.loads(Path(path).read_text())

    # --- model source (for APPLY / REPAIR) ----------------------------------
    def model_root(self) -> Path:
        return Path(self.manifest["config"]["model_root"])

    def model_files(self) -> list[Path]:
        root = self.model_root()
        return [root / rel for rel in self.manifest.get("pathmap", {}).get("model_files", [])]

    # --- telemetry (PLAN section 10.1) --------------------------------------
    def record_agent_call(
        self,
        stage: str,
        role: str,
        model: str,
        usage: dict | None,
        prompt: str | None = None,
        response: str | None = None,
    ) -> None:
        """One row per query(); accumulate cost+tokens into state (budget gate input).

        agent_calls.jsonl is the LEAN cost/token stream. When prompt/response
        are given, the full payload is appended to a single prompts.jsonl,
        linked 1:1 by agent_call_id (no per-call files).
        """
        import hashlib

        from .events import append_jsonl, make_agent_call_row, next_agent_call_seq

        usage = usage or {}
        ts = utc_ts()
        iteration = self.state.get("iteration")
        seq = next_agent_call_seq(self._agent_calls)  # monotonic, unique per run
        prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()[:12] if prompt else None
        response_sha = hashlib.sha256(response.encode()).hexdigest()[:12] if response else None
        row = make_agent_call_row(
            run_id=self.run.run_id,
            phase="loop",
            iteration=iteration,
            stage=stage,
            role=role,
            model=model,
            usage=usage,
            seq=seq,
            prompt_sha=prompt_sha,
            response_sha=response_sha,
            ts=ts,
        )
        if prompt is not None or response is not None:
            append_jsonl(
                self.run.dir / "prompts.jsonl",
                {
                    "agent_call_id": row["agent_call_id"],  # FK -> agent_calls.jsonl
                    "run_id": self.run.run_id,
                    "ts": ts,
                    "phase": "loop",
                    "iteration": iteration,
                    "stage": stage,
                    "role": role,
                    "model": model,
                    "prompt": prompt or "",
                    "response": response or "",
                    "prompt_sha": prompt_sha,
                    "response_sha": response_sha,
                },
            )
        append_jsonl(self._agent_calls, row)
        self.state["cost_usd"] = round(self.state.get("cost_usd", 0.0) + (usage.get("cost_usd") or 0.0), 6)
        self.state["tokens_in"] = self.state.get("tokens_in", 0) + (usage.get("tokens_in") or 0)
        self.state["tokens_out"] = self.state.get("tokens_out", 0) + (usage.get("tokens_out") or 0)

    def log_event(self, stage: str, status: str, detail: str = "") -> None:
        write_event(
            self._events,
            phase="loop",
            stage=stage,
            event=status,
            detail=detail,
            iteration=self.state.get("iteration"),
        )
