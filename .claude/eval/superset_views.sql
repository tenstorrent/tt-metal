-- SQL views for Apache Superset dashboards.
--
-- These views provide pre-aggregated data that Superset charts consume directly.
-- Run once against your PostgreSQL (or SQLite) eval database.
--
-- Usage:
--   psql -d eval -f eval/superset_views.sql
--   # or for SQLite:
--   sqlite3 eval_runs.db < eval/superset_views.sql

-- ---------------------------------------------------------------------------
-- 1. Runs overview — main table for the Superset runs dashboard
-- ---------------------------------------------------------------------------
-- Includes a computed detail_url column for linking to the Flask detail server.
CREATE VIEW IF NOT EXISTS v_runs_overview AS
SELECT
    r.id,
    r.timestamp,
    r.prompt_name,
    r.run_number,
    r.starting_branch,
    r.created_branch,
    r.status,
    r.phase,
    r.score_total,
    r.score_grade,
    r.golden_passed,
    r.golden_total,
    CASE
        WHEN r.golden_total > 0 THEN ROUND(r.golden_passed * 100.0 / r.golden_total, 1)
        ELSE NULL
    END AS golden_pass_pct,
    r.duration_seconds,
    CASE
        WHEN r.duration_seconds IS NOT NULL THEN r.duration_seconds / 60
        ELSE NULL
    END AS duration_minutes,
    r.annotation_score,
    r.golden_name,
    '/run/' || r.id AS detail_url
FROM runs r
ORDER BY r.timestamp DESC;


-- ---------------------------------------------------------------------------
-- 2. Failure summary — for pie/bar charts of failure categories
-- ---------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS v_failure_summary AS
SELECT
    r.id AS run_id,
    r.prompt_name,
    r.golden_name,
    r.timestamp,
    t.failure_category,
    COUNT(*) AS failure_count
FROM test_results t
JOIN runs r ON t.run_id = r.id
WHERE t.status != 'passed' AND t.failure_category IS NOT NULL
GROUP BY r.id, r.prompt_name, r.golden_name, r.timestamp, t.failure_category;


-- ---------------------------------------------------------------------------
-- 3. Score trends — for line charts over time
-- ---------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS v_score_trends AS
SELECT
    r.timestamp,
    r.prompt_name,
    r.golden_name,
    r.run_number,
    r.score_total,
    r.score_grade,
    r.golden_passed,
    r.golden_total,
    CASE
        WHEN r.golden_total > 0 THEN ROUND(r.golden_passed * 100.0 / r.golden_total, 1)
        ELSE NULL
    END AS golden_pass_pct,
    r.duration_seconds
FROM runs r
WHERE r.status = 'complete' AND r.score_total IS NOT NULL
ORDER BY r.timestamp;


-- ---------------------------------------------------------------------------
-- 4. Per-operation stats — aggregated metrics per operation
-- ---------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS v_per_op_stats AS
SELECT
    COALESCE(r.golden_name, r.prompt_name) AS operation,
    COUNT(*) AS total_runs,
    ROUND(AVG(r.score_total), 1) AS avg_score,
    MIN(r.score_total) AS min_score,
    MAX(r.score_total) AS max_score,
    SUM(CASE WHEN r.golden_passed = r.golden_total AND r.golden_total > 0 THEN 1 ELSE 0 END) AS full_pass_count,
    CASE
        WHEN COUNT(*) > 0 THEN
            ROUND(SUM(CASE WHEN r.golden_passed = r.golden_total AND r.golden_total > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1)
        ELSE 0
    END AS full_pass_rate,
    ROUND(AVG(r.duration_seconds), 0) AS avg_duration_seconds
FROM runs r
WHERE r.status = 'complete'
GROUP BY COALESCE(r.golden_name, r.prompt_name);


-- ---------------------------------------------------------------------------
-- 5. Active runs — for the "currently running" widget
-- ---------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS v_active_runs AS
SELECT
    r.id,
    r.timestamp,
    r.prompt_name,
    r.status,
    r.phase,
    r.created_branch,
    '/run/' || r.id AS detail_url
FROM runs r
WHERE r.status NOT IN ('complete', 'failed')
ORDER BY r.timestamp DESC;


-- ---------------------------------------------------------------------------
-- 6. Test results detail — for drill-down from failure charts
-- ---------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS v_test_results AS
SELECT
    t.id,
    t.run_id,
    r.prompt_name,
    r.golden_name,
    r.timestamp AS run_timestamp,
    t.test_name,
    t.test_file,
    t.shape,
    t.status,
    t.failure_category,
    t.failure_message
FROM test_results t
JOIN runs r ON t.run_id = r.id
ORDER BY r.timestamp DESC, t.id;


-- ---------------------------------------------------------------------------
-- 7. Grade distribution — for bar/pie chart of score grades
-- ---------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS v_grade_distribution AS
SELECT
    r.score_grade AS grade,
    COUNT(*) AS count
FROM runs r
WHERE r.score_grade IS NOT NULL
GROUP BY r.score_grade;
