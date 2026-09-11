-- Synthetic test schema snapshot from tt_ops_code_gen eval/db.py at
-- 034527ad845a7b61596139802c4af567983a14bb; never used against the remote DB.
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    prompt_name TEXT NOT NULL,
    run_number INTEGER NOT NULL,
    starting_branch TEXT NOT NULL,
    starting_commit TEXT NOT NULL,
    created_branch TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    phase TEXT,
    score_total REAL,
    score_grade TEXT,
    golden_passed INTEGER,
    golden_total INTEGER,
    annotation_score INTEGER,
    annotation_notes TEXT,
    golden_name TEXT,
    duration_seconds INTEGER,
    total_cost_usd REAL,
    total_turns INTEGER,
    total_input_tokens INTEGER,
    total_output_tokens INTEGER,
    model TEXT,
    failure_reason TEXT,
    agent TEXT NOT NULL DEFAULT 'claude',
    device_wait_seconds REAL,
    device_run_seconds REAL,
    device_invocations INTEGER,
    target_spec TEXT,
    op_metadata_json TEXT,
    run_source TEXT NOT NULL DEFAULT 'local',
    runtime_backend TEXT NOT NULL DEFAULT 'hw',
    run_host TEXT,
    run_user TEXT,
    eval_branch TEXT,
    eval_commit TEXT,
    arch TEXT,
    effort TEXT
);

CREATE TABLE IF NOT EXISTS test_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    test_name TEXT NOT NULL,
    test_file TEXT,
    shape TEXT,
    status TEXT NOT NULL,
    failure_category TEXT,
    failure_message TEXT,
    phase TEXT,
    pcc REAL,
    rms REAL,
    max_abs_diff REAL,
    median_abs_diff REAL,
    ulp_p99 REAL,
    device_kernel_ns REAL,
    device_num_cores INTEGER,
    axes_json TEXT,
    extras_json TEXT
);

CREATE TABLE IF NOT EXISTS score_criteria (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    criterion TEXT NOT NULL,
    raw_score REAL NOT NULL,
    weight REAL NOT NULL,
    weighted_score REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS kernels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    filename TEXT NOT NULL,
    source_code TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS host_code (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    filename TEXT NOT NULL,
    source_code TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    name TEXT NOT NULL,
    content TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kw_breadcrumbs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    agent_name TEXT NOT NULL,
    content TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS refinement_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    phase TEXT NOT NULL,
    golden_passed INTEGER NOT NULL,
    golden_total INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    supported_pass INTEGER,
    supported_fail INTEGER,
    supported_marked_xfail INTEGER,
    xfail_expected INTEGER,
    xpass_drift INTEGER,
    xfail_wrong_mode INTEGER,
    invalid_skipped INTEGER,
    invalid_unexpected INTEGER,
    infeasible_skipped INTEGER,
    no_axes_found INTEGER,
    registry_snapshot TEXT,
    translated_hang_nodeid TEXT,
    refinement_type TEXT
);

CREATE TABLE IF NOT EXISTS phases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    name TEXT NOT NULL,
    ordering INTEGER NOT NULL,
    exit_code INTEGER,
    subtype TEXT,
    stop_reason TEXT,
    num_turns INTEGER,
    max_turns INTEGER,
    duration_ms INTEGER,
    cost_usd REAL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cache_read_tokens INTEGER,
    cache_creation_tokens INTEGER,
    model TEXT
);

CREATE TABLE IF NOT EXISTS device_timings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    source TEXT NOT NULL,
    pid INTEGER,
    started_at_ms INTEGER NOT NULL,
    wait_ms INTEGER NOT NULL,
    run_ms INTEGER NOT NULL,
    test_path TEXT,
    exit_code INTEGER,
    phase TEXT,
    precompile_mode TEXT,
    precompile_reason TEXT,
    precompile_programs INTEGER
);

CREATE TABLE IF NOT EXISTS device_phases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    phase TEXT NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER,
    ordering INTEGER
);

CREATE INDEX IF NOT EXISTS idx_test_results_run_phase ON test_results(run_id, phase);
CREATE INDEX IF NOT EXISTS idx_test_results_failure_category ON test_results(failure_category);
CREATE INDEX IF NOT EXISTS idx_score_criteria_run ON score_criteria(run_id);
CREATE INDEX IF NOT EXISTS idx_kernels_run ON kernels(run_id);
CREATE INDEX IF NOT EXISTS idx_host_code_run ON host_code(run_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id);
CREATE INDEX IF NOT EXISTS idx_kw_breadcrumbs_run ON kw_breadcrumbs(run_id);
CREATE INDEX IF NOT EXISTS idx_refinement_snapshots_run ON refinement_snapshots(run_id);
CREATE INDEX IF NOT EXISTS idx_phases_run ON phases(run_id);
CREATE INDEX IF NOT EXISTS idx_device_timings_run ON device_timings(run_id);
CREATE INDEX IF NOT EXISTS idx_device_phases_run ON device_phases(run_id);
