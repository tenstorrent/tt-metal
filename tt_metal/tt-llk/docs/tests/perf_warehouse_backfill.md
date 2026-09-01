# Sending nightly perf data to the warehouse

The LLK perf nightly already writes one typed Parquet per shard. This page
tells you how to move those Parquets into the perf warehouse over SFTP, and how
to check afterwards that they arrived complete.

Use it for two jobs:

- **Backfill** — load nights that ran before the warehouse existed.
- **Rehearsal** — prove the naming and the SFTP leg on real data, before the
  same upload runs inside CI.

## Where the data comes from

`.github/workflows/llk-perf.yaml` runs the nightly at 03:00 UTC. Each shard
writes `perf_data/runs/<tag>/<tag>.parquet` and uploads it as the artifact
`perf-data-<arch>-<split_group>`. The run tag is
`<run_id>-<arch>-<job_index>`.

The tool takes those artifacts. It does not run tests and does not need a chip.

## Prerequisites

1. `gh auth login`, for the artifact download.
2. `pandas` and `pyarrow` (both are in `tests/requirements.txt`).
3. The SFTP host, the user name, and **your private key** — the key whose
   `.pub` half the data team authorised. Never pass the `.pub` file.

## Run it

All paths below are relative to `tests/python_tests/`.

```bash
# 1. Rehearse. Stage the last 5 nightlies and print the sftp command. No upload.
python3 -m helpers.perf.backfill --last 5 --stage-dir /tmp/stage --dry-run \
    --host "$LLK_PERF_SFTP_HOST" --user llk-perf-run --key ~/.ssh/id_ed25519

# 2. Upload named runs for real.
python3 -m helpers.perf.backfill \
    --run-id 33145544147 --run-id 33230616532 --run-id 33465181016 \
    --stage-dir /tmp/stage --upload \
    --host "$LLK_PERF_SFTP_HOST" --user llk-perf-run --key ~/.ssh/id_ed25519

# 3. Re-upload what is already staged. Needs no GitHub access.
python3 -m helpers.perf.backfill --from-dir /tmp/stage/runs \
    --stage-dir /tmp/stage --upload ...
```

`--host`, `--user` and `--key` also read `LLK_PERF_SFTP_HOST`,
`LLK_PERF_SFTP_USER` and `LLK_PERF_SFTP_KEY`.

`--last N` counts **successful scheduled** runs only. A `workflow_dispatch` run
is somebody testing, and it must not enter the baseline history.

## What lands on the server

One flat object per shard, in the SFTP home directory:

```
llk_perf_<run_id>-<arch>-<job_index>.parquet
```

Flat, because the run tag is already unique across nights. Flat also means no
`mkdir` on the server and no ordering rules between uploads, which is what a
Snowflake external stage wants.

The name comes from the **rows**, not from the file name on disk. The two differ
after a re-run: the workflow builds the tag from `github.run_id`, which drops
the attempt number, but `core._run_id` writes `<run_id>-<attempt>` into the
rows. Attempt 2 therefore lands as `llk_perf_<run_id>-2-<arch>-<n>.parquet` and
does not overwrite attempt 1.

## What is refused

A file is skipped, and the reason is printed, when it:

- does not read as Parquet, or holds no rows;
- misses a mandatory column, or has NULL in one;
- has a non-constant `arch` or `run_id`; or
- has a name that disagrees with its rows.

One bad shard never blocks the other nine.

Runs older than the in-run Parquet writer hold CSVs only. The tool reports them
as skipped. They are outside what the warehouse can take.

## Checking the ingest

Every upload writes `manifest.csv` into the stage directory:

| column | meaning |
|---|---|
| `file` | the object name on the server |
| `tag` | the run tag on disk |
| `arch`, `run_id`, `commit_sha`, `timestamp`, `pipeline` | row provenance |
| `rows` | rows in that shard |
| `bytes` | size of the staged copy |

The manifest is the checklist for the read-back. After the ingest, every row in
it must appear in the warehouse with the same row count.

You can also read the staged files directly, before the warehouse has them:

```sql
-- duckdb
SELECT run_id, arch, min(timestamp)[1:10] AS night, count(*) AS rows
FROM read_parquet('/tmp/stage/*.parquet', union_by_name=true)
GROUP BY 1, 2 ORDER BY night;
```

## Tests

`test_perf_backfill.py` covers the tool. It runs offline: no chip, no GitHub,
no SFTP server. `--from-dir` and `--dry-run` are what make that possible, so
keep them working.
