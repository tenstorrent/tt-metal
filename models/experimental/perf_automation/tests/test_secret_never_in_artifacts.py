"""PLAN section 3.1: the credential never appears in any produced artifact.

Deferred from M0 (no artifact producers yet); M1 built them (Checkpoint,
Ledger, Manifest), so it runs now: load a fake .env.agent, write a full set
of run artifacts, then grep every produced file for the key value -> 0 hits.
"""

from agent.checkpoint import Checkpoint
from agent.config import apply_agent_env
from agent.ledger import Ledger
from agent.run import Run

SECRET = "sk-super-secret-DO-NOT-LEAK-0123456789"


def test_secret_never_in_artifacts(tmp_path):
    env_file = tmp_path / ".env.agent"
    env_file.write_text(f"LITELLM_BASE_URL=https://proxy.example\nLITELLM_API_KEY={SECRET}\n")

    # Key is loaded into the SDK process env ONLY (never into artifacts).
    sdk_env: dict[str, str] = {}
    apply_agent_env(env_file, sdk_env)
    assert sdk_env["ANTHROPIC_API_KEY"] == SECRET

    runs_root = tmp_path / "runs"
    run = Run.create(
        runs_root,
        config={"metric": {"name": "wall_ms", "direction": "min", "target": 12.0}},
        run_id="2026-06-09T14-22",
    )
    Checkpoint(run.state_path).save({"run_id": run.run_id, "state": "ROUTE", "iteration": 1, "cost_usd": 0.42})
    Ledger(run.ledger_path).append(
        {
            "experiment_id": f"{run.run_id}#1",
            "lever": "mlp-fidelity-walk",
            "status": "baseline",
            "hypothesis": "start from baseline",
        }
    )
    (run.profiles_dir / "iter_00_baseline.csv").write_text("OP CODE,DEVICE KERNEL DURATION [ns]\nMatmul,123\n")

    # Grep every produced artifact for the secret value.
    hits = [str(p) for p in runs_root.rglob("*") if p.is_file() and SECRET.encode() in p.read_bytes()]
    assert hits == [], f"secret leaked into: {hits}"
