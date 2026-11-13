# fabric_debug_tool

## Overview

When a fabric workload hangs, the state of its routers is still on the chips. `fabric_debug_tool.py` saves that state
and makes sense of it, for every rank of the workload, with one command:

1. **Capture:** it runs `fabric_capture` on every rank. Each one reads its rank's fabric routers off the chips and saves
   the bytes it read. See capture's [README](capture/README.md).
2. **Decode:** it then runs `fabric_decode` on every rank. Each one turns its rank's bytes into values, using the
   rank's manifest, and writes a decoded file. Decode doesn't touch the chips.
3. **Report:** it prints where the files are, and the command that copies them somewhere else.

To reach every rank, the tool reuses the `tt-run` command the workload was launched with. You give it that command,
minus the program at the end. The tool adds `fabric_capture` to the end and runs it, then does the same with
`fabric_decode`. Each tool process then runs on the same host, as the same rank, with the same environment as the
workload's rank did, so it finds the rank's files and chips by itself.

Every rank is captured under one capture id, which is the UTC time the tool started unless you choose one.

## Before you run it

The tool assumes the following. If any of them doesn't hold, its results can be wrong or missing.

- **The workload wrote its manifests.** It was run with `TT_METAL_FABRIC_GENERATE_MANIFEST=1`, so each rank wrote a
  manifest when fabric started. The manifest tells capture which chips and routers the rank owns and where to read.
- **You run it right after the hang, and nothing else has used the machine since.** Nothing has reset the chips, run
  on them or rediscovered them. A reset wipes the state the tool is there to read. The hung workload can still be
  running: capture opens the chips alongside it, and only reads.
- **The files the workload's `tt-run` command used still exist, unchanged.** That means every file the command names,
  such as the rank bindings file or rank bindings mapping, the mesh graph descriptor and any host file. It also means
  the rank bindings `tt-run` generated for the workload, when it found the rank bindings itself: `tt-run` keeps those
  under `generated/ttrun/` in the folder it was started from, and reuses them on a later run with the same
  descriptor and hosts.
- **You start it from the folder you started the workload's `tt-run` from.** `tt-run` looks for its generated files
  in that folder, and relative paths in the command are relative to it. If `tt-run` doesn't find its generated rank
  bindings there, it rediscovers the machine, which touches the hung chips and may not place the ranks as before.
- **The tool's ranks see the workload's logs folder.** Each rank looks for its files under `TT_METAL_LOGS_PATH`, else
  its working folder. Set `TT_METAL_LOGS_PATH` to what the workload had. `tt-run` passes it on to every rank.
- **The tools are from the same build as the workload.** The tools refuse a manifest that another build wrote.
- **The program is the last thing in the workload's command.** Everything before it is what you give the tool.

## Running it

Give the tool its own options, then `--`, then the workload's `tt-run` command without the program:

```bash
# The workload was launched with:
#   tt-run --rank-binding rank_bindings.yaml --mpi-args "--tag-output" python3 my_workload.py
# Once it hangs, from the same folder:
export TT_METAL_LOGS_PATH=<the workload's logs folder>
python3 $TT_METAL_HOME/tt_metal/fabric/debug/visualizer/fabric_debug_tool.py \
    -- tt-run --rank-binding rank_bindings.yaml --mpi-args "--tag-output"
```

For a workload that wasn't launched with `tt-run`, a single process, leave out the `--` and the command. The tool then
runs `fabric_capture` and `fabric_decode` itself, once each:

```bash
python3 $TT_METAL_HOME/tt_metal/fabric/debug/visualizer/fabric_debug_tool.py
```

| Option | Default | What it does |
| --- | --- | --- |
| `--capture-id <id>` | The UTC time now, such as `20261105T221500Z` | Names the capture's folder in each run folder. It can't contain `/` |
| `--rounds <n>` | `fabric_capture`'s default, 3 | Passed to `fabric_capture`: how many times it reads every router's multi-sampled regions before reading each core in full |
| `--round-interval-ms <ms>` | `fabric_capture`'s default, 1000 | Passed to `fabric_capture`: the time between rounds |
| `--build-dir <dir>` | `$TT_METAL_HOME/build` | The build whose `tools/fabric_capture` and `tools/fabric_decode` it runs. Without it, `TT_METAL_HOME` must be set |
| `-- <command>` | None: the tools run as one process | The workload's `tt-run` command without the program |

The tool checks its options, and that both tools are in the build, before it runs anything. It won't run a command
that has `--force-rediscovery`, since that would rediscover the hung chips. It warns when `TT_METAL_LOGS_PATH` isn't
set.

## What it leaves

Each rank's files go in its fabric run's folder, beside the manifest the workload wrote. A workload split into
sub-contexts has one run folder per sub-context.

```
<logs>/generated/fabric_debug/runs/<run_id>/
  fabric_manifest_rank_<r>_of_<n>.json      one per rank, written by the workload
  captures/<capture_id>/
    fabric_capture_rank_<r>_of_<n>.json     what capture read, and from where
    fabric_capture_rank_<r>_of_<n>.bin      the bytes it read
    fabric_decoded_rank_<r>_of_<n>.json     what decode made of them
```

When both steps succeed, the tool prints the capture folders it can see from its host, and the command that copies
them, such as:

```
fabric_debug_tool: capture and decode successful, capture id 20261105T223000Z
Artifacts stored in:
  /data/me/logs/generated/fabric_debug/runs/20261105T221500Z-1a2b3c4d/captures/20261105T223000Z
To copy them to /scratch/$USER/fabric_debug_tool_artifacts/20261105T223000Z/ (or another folder with --dest):
  python3 .../fabric_debug_tool_copy.py 20261105T223000Z --logs-path /data/me/logs
```

When the ranks ran on other hosts and the logs folder isn't shared between them, each host has its own ranks' files.

## When something fails

The tool stops at the first step that fails, and exits with that step's exit status:

| Exit status | Meaning |
| --- | --- |
| 0 | Capture and decode both succeeded on every rank |
| 1 | A rank's capture or decode wrote its files, but something is missing from them. Capture's record, or decode's findings, say what |
| 2 | The tool's own options or setup are wrong, or a capture or decode failed and wrote nothing |
| 126, 127 | The command couldn't be started: it can't be run, or isn't there |
| 128 + a signal | A step was killed by that signal |
| Anything else | `tt-run`'s own exit status |

With `tt-run`, the status is `tt-run`'s, which it takes from the ranks. When one rank's tool exits non-zero, MPI may
stop the other ranks, so their files may be missing too. A capture that stopped with status 1 still wrote its files:
you can decode them by running `fabric_decode --capture-id <id>` through the same `tt-run` command. Run
`fabric_decode --help` for its options.

## Copying the results

`fabric_debug_tool_copy.py` copies one capture, and the manifests it was taken from, to a folder of their own. Run it
where the copy should go, with the command `fabric_debug_tool.py` printed:

```bash
python3 $TT_METAL_HOME/tt_metal/fabric/debug/visualizer/fabric_debug_tool_copy.py <capture_id> \
    [--logs-path <path>] [--dest <dir>]
```

| Option | Default | What it does |
| --- | --- | --- |
| `<capture_id>` | Required | The capture to copy |
| `--logs-path <path>` | `TT_METAL_LOGS_PATH`, else the working folder | The workload's logs folder |
| `--dest <dir>` | `/scratch/$USER/fabric_debug_tool_artifacts` | Where the capture's folder goes |

For each run folder that has the capture, it copies the run's manifests and the capture's folder, and nothing else:

```
<dest>/<capture_id>/
  <run_id>/
    fabric_manifest_rank_<r>_of_<n>.json
    captures/<capture_id>/
      fabric_capture_rank_<r>_of_<n>.json, .bin
      fabric_decoded_rank_<r>_of_<n>.json
```

It only reads the logs folder. It copies into a temporary folder first and moves it into place at the end, so a copy
that fails leaves nothing behind. It exits 0 when it copied the capture, and 2 when it copied nothing: no run has the
capture, `<dest>/<capture_id>` already exists, or a file couldn't be copied. The message says which.

## Running the steps yourself

Capture and decode also run on their own, for example to decode a capture again, or to capture without decoding.
Launch them through the workload's `tt-run` command as the tool does, with the same `--capture-id` for every rank:

```bash
tt-run <the workload's options> $TT_METAL_HOME/build/tools/fabric_capture --capture-id <id>
tt-run <the workload's options> $TT_METAL_HOME/build/tools/fabric_decode --capture-id <id>
```

See capture's [README](capture/README.md) for its options and output, and `fabric_decode --help` for decode's.
