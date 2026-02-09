-- TTNN Sweeps Slack Notification Database Schema

-- Table 1: Run metadata
CREATE TABLE IF NOT EXISTS sweep_run (
    run_id SERIAL PRIMARY KEY,
    github_pipeline_id BIGINT UNIQUE,
    run_contents TEXT NOT NULL,        -- 'lead models', 'nightly', etc.
    card_type TEXT NOT NULL,           -- 'wormhole_b0', 'blackhole'
    git_sha TEXT,
    git_branch TEXT,
    run_start_ts TIMESTAMPTZ NOT NULL,
    test_count INT,
    pass_count INT,
    fail_count INT
);

CREATE INDEX IF NOT EXISTS sweep_run_lookup_idx
    ON sweep_run (run_contents, card_type, run_start_ts DESC);

-- Table 2: Test results (denormalized - includes perf metric inline)
CREATE TABLE IF NOT EXISTS sweep_test (
    test_id SERIAL PRIMARY KEY,
    run_id INT REFERENCES sweep_run(run_id) ON DELETE CASCADE,
    full_test_name TEXT NOT NULL,
    input_hash TEXT,                   -- Hash of test input parameters for cross-run matching
    op_name TEXT,
    model_name TEXT,
    status TEXT NOT NULL,              -- 'pass', 'fail_assert_exception', etc.
    device_fw_duration_ns FLOAT    -- NULL if not measured
);

CREATE INDEX IF NOT EXISTS sweep_test_run_idx ON sweep_test (run_id);
CREATE INDEX IF NOT EXISTS sweep_test_status_idx ON sweep_test (run_id, status);
CREATE INDEX IF NOT EXISTS sweep_test_match_idx ON sweep_test (full_test_name, input_hash);
