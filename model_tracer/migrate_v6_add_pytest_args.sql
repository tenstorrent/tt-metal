-- Migration: add pytest_args column to ttnn_ops_v6.trace_run
--
-- Why: lets us answer "have I traced model X with these pytest args on
--      hardware Y?" without re-running the tracer. pytest_args is the
--      string of CLI flags passed after `--` to generic_ops_tracer.py,
--      e.g. `-k "4x8sp0tp1 and encoder_device" --tb=short`.
--
-- Safe to run on existing deployments: the column is nullable, so all
-- pre-migration trace_run rows simply have pytest_args = NULL.
--
-- Idempotent: ALTER ... ADD COLUMN IF NOT EXISTS / CREATE INDEX IF NOT EXISTS.

ALTER TABLE ttnn_ops_v6.trace_run
    ADD COLUMN IF NOT EXISTS pytest_args TEXT;

CREATE INDEX IF NOT EXISTS trace_run_pytest_args_idx
    ON ttnn_ops_v6.trace_run(pytest_args)
    WHERE pytest_args IS NOT NULL;
