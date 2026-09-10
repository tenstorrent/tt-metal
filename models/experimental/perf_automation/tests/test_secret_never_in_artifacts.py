"""PLAN section 3.1: the credential never appears in any produced artifact.

Native auth: ANTHROPIC_API_KEY comes from the shell env (no LiteLLM proxy file). It must reach the
SDK process env ONLY — never any run artifact: write a full set (Checkpoint, Ledger, Manifest, CSV)
then grep every produced file for the key value -> 0 hits."""

from agent.checkpoint import Checkpoint
from agent.config import load_agent_env
from agent.ledger import Ledger
from agent.run import Run

SECRET = "sk-super-secret-DO-NOT-LEAK-0123456789"


def test_secret_never_in_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)

    resolved = load_agent_env(tmp_path / ".env.agent")
    assert resolved["ANTHROPIC_API_KEY"] == SECRET

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
