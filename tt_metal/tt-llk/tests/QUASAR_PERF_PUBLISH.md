# Publishing Quasar perf runs to the warehouse

Quasar has no CI perf runner. People run the perf suite by hand on the
emulator. This page shows how to send such a run to the LLK perf warehouse
(Snowflake `LLK_PERF`), where the dashboard reads it.

The tool is `helpers/perf/publish_manual.py`. Every file it writes has
`pipeline = "manual"` and a run id like
`manual-20261005-20261005T143012Z-quasar`. The run id comes from the run's
start time. If you publish the same run again, the new copy replaces the old one.

Every file also records what executed the run: `platform` (`emulator`,
`simulator` or `silicon`) and `platform_version` (its build, for example the
emulator image). `platform` is mandatory. Numbers from two platforms must never
share a trend.

## Who can upload

Only the people whose public key `data_iac` lists for the `llk-perf-run` SFTP
user can upload. To get access, make an SSH key pair on the machine you publish
from, and send the public key to the data team. Without a listed key, the tool
writes the Parquet file and does not send it. Ask a listed person to send it.

## Publish a run you just made

1. Run the perf tests from a clean checkout of a commit on `main`.
2. From `tests/python_tests`, write the file and read its output:

   ```bash
   python -m helpers.perf.publish_manual publish \
       --platform emulator --platform-version <build>
   ```

   The default input is `perf_data/latest`, which is the last run. Use
   `--run-dir` for another run.
3. Send it:

   ```bash
   python -m helpers.perf.publish_manual publish \
       --platform emulator --platform-version <build> \
       --upload --key ~/.ssh/<key>
   ```

   You can also set `LLK_PERF_SFTP_KEY` instead of `--key`.

The tool does not upload if the checkout has uncommitted changes or if `HEAD` is
not on `origin/main`. Those numbers would show in the trend as a change on
`main`. If you made the run from another checkout, give `--commit <sha>`.

The conversion is strict. If a CSV has a column that
`helpers/perf/wide_schema_quasar.py` does not know, the publish fails. Add the
column to that schema.

## Backfill old runs

1. Put each old run in its own directory, with its per-test CSVs inside.
2. Add a `run_meta.json` to each directory:

   ```json
   {
     "timestamp": "2026-08-14T09:00:00Z",
     "commit_sha": "<sha of the run>",
     "platform": "emulator",
     "platform_version": "<build>"
   }
   ```

   `timestamp` is mandatory. It sets the run id and the run's position in the
   trend. Without `commit_sha`, the run loads with commit `unknown`. If all
   runs used the same platform, you can give `--platform` and
   `--platform-version` to `backfill` instead.
3. Convert:

   ```bash
   python -m helpers.perf.publish_manual backfill --archive <dir> --out-dir <out>
   ```

4. Read the report. A dropped column is data that the warehouse does not get.
5. Send the files:

   ```bash
   python -m helpers.perf.publish_manual upload <out>/*.parquet --key ~/.ssh/<key>
   ```

Use `--dry-run` on `upload` (or on `publish --upload`) to see the SFTP commands
before you send.
