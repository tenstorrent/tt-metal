-- Migration: add trace-time software provenance columns to ttnn_ops_v6.trace_run
--
-- Why: capture tt-kmd / tt-smi / tt-firmware versions from tt-smi at the
-- time the trace was collected. This preserves machine software state per
-- trace_run so downstream analysis can correlate behavior with environment.
--
-- Safe on existing deployments: all columns are nullable.
-- Idempotent: ADD COLUMN IF NOT EXISTS / CREATE INDEX IF NOT EXISTS.

ALTER TABLE ttnn_ops_v6.trace_run
    ADD COLUMN IF NOT EXISTS tt_kmd TEXT,
    ADD COLUMN IF NOT EXISTS tt_smi TEXT,
    ADD COLUMN IF NOT EXISTS tt_firmware TEXT;

CREATE INDEX IF NOT EXISTS trace_run_tt_kmd_idx
    ON ttnn_ops_v6.trace_run(tt_kmd)
    WHERE tt_kmd IS NOT NULL;

CREATE INDEX IF NOT EXISTS trace_run_tt_smi_idx
    ON ttnn_ops_v6.trace_run(tt_smi)
    WHERE tt_smi IS NOT NULL;

CREATE INDEX IF NOT EXISTS trace_run_tt_firmware_idx
    ON ttnn_ops_v6.trace_run(tt_firmware)
    WHERE tt_firmware IS NOT NULL;
