# dep-graph — runbook

Self-contained guide for re-running the tt-metal polyglot dependency-graph
pipeline. Read this if you want to rebuild `dep-graph.sqlite` without me.

The companion spec / decision log is `/home/ubuntu/tt-metal/opus-instructions.md`.
This file is intentionally narrower: just how to run the thing.

## TL;DR — full-repo run from scratch

```bash
# All commands run from /home/ubuntu/tt-metal on the host.
# Docker on this host requires sudo.

# 0) Make sure the container is up. It bind-mounts this repo to /workspace.
sudo docker ps | grep tt-metal-basic-dev-container || ./create-docker-container.sh

# 1) (one-time) generate compile_commands.json inside the container
sudo docker exec --user 1000:1000 tt-metal-basic-dev-container bash -c '
  source /opt/venv/bin/activate &&
  cd /workspace/build_Release &&
  cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON .
'

# 2) Index all in-scope C++ TUs in parallel. ~30 min on 14 cores, fully cached.
sudo docker exec --user 1000:1000 tt-metal-basic-dev-container bash -c '
  source /opt/venv/bin/activate &&
  python /workspace/dep-graph/scripts/cpp_index_driver.py \
      --db /workspace/build_Release/compile_commands.json \
      --workers 14
'

# 3) Index all Python under ttnn/ttnn/ (a few seconds)
sudo docker exec --user 1000:1000 tt-metal-basic-dev-container bash -c '
  source /opt/venv/bin/activate &&
  python /workspace/dep-graph/scripts/py_index.py \
      --dir /workspace/ttnn/ttnn \
      --module-root /workspace/ttnn \
      --out /workspace/dep-graph/cache/py_index.json
'

# 4) Fold per-TU shards into one merged index
sudo docker exec --user 1000:1000 tt-metal-basic-dev-container bash -c '
  source /opt/venv/bin/activate &&
  python /workspace/dep-graph/scripts/cpp_index_merger.py
'

# 5) Build the SQLite DB
sudo docker exec --user 1000:1000 tt-metal-basic-dev-container bash -c '
  source /opt/venv/bin/activate &&
  python /workspace/dep-graph/scripts/stitch_sqlite.py
'

# 6) Validate
sudo docker exec --user 1000:1000 tt-metal-basic-dev-container bash -c '
  source /opt/venv/bin/activate &&
  python /workspace/dep-graph/scripts/validate.py
'
```

Output: `dep-graph/out/dep-graph.sqlite`. Open with any sqlite3 client; query
through `dep-graph/scripts/query.py` for canned questions.

---

## Prerequisites

- **Docker container `tt-metal-basic-dev-container`** running.
  Recreate with `./create-docker-container.sh` if it's gone. It bind-mounts
  the repo to `/workspace`. Docker on this host requires `sudo`.
- **`--user 1000:1000` on every `docker exec`.** Otherwise the container
  writes root-owned files into the bind mount and corrupts ownership.
- **Python virtualenv inside the container at `/opt/venv`.** All `python`
  invocations must source `/opt/venv/bin/activate` first (or use
  `/opt/venv/bin/python` directly).
- **One-time pip installs** inside the venv:
  ```bash
  sudo docker exec --user 1000:1000 -e UV_CACHE_DIR=/tmp/uv-cache \
      tt-metal-basic-dev-container bash -c '
        source /opt/venv/bin/activate && uv pip install libclang pyyaml
      '
  ```
  `libclang` (Python bindings, 18.1.1) and `pyyaml` (validate.py).
- **`compile_commands.json`** at `/workspace/build_Release/compile_commands.json`.
  See TL;DR step 1 if missing.

## Pipeline architecture

```
┌──────────────────────┐
│ compile_commands.json│ (CMake)
└────────┬─────────────┘
         ↓
┌──────────────────────┐    ┌──────────────────────┐
│ cpp_index_driver.py  │←──→│ tu_shards/<hash>/    │ ← per-TU JSONL cache,
│  (multiprocess pool) │    │  manifest.json       │   mtime-keyed for
│  spawns workers      │    │  nodes.jsonl         │   incremental runs
└────────┬─────────────┘    │  edges.jsonl         │
         │                  │  bindings.jsonl      │
         ↓                  │  diagnostics.jsonl   │
┌──────────────────────┐    └──────────────────────┘
│ cpp_index_worker.py  │             ↓
│  (one TU per process)│    ┌──────────────────────┐
└──────────────────────┘    │ cpp_index_merger.py  │
                            │  USR-keyed dedup     │
                            └────────┬─────────────┘
                                     ↓
                            ┌──────────────────────┐
                            │ cpp_index/           │ ← consolidated JSONL
                            │  nodes.jsonl         │
                            │  edges.jsonl         │
                            │  bindings.jsonl      │
                            └────────┬─────────────┘
                                     │
┌──────────────────────┐             │
│ py_index.py          │─→ py_index.json
│  (ast, single thread)│             │
└────────┬─────────────┘             │
         └─────────────┬─────────────┘
                       ↓
              ┌──────────────────┐
              │ stitch_sqlite.py │ ← cross-language resolution:
              └────────┬─────────┘    py_registrations preferred,
                       ↓               bindings fallback
              ┌──────────────────┐
              │ dep-graph.sqlite │ ← the queryable artifact
              └────────┬─────────┘
                       │
                       ├─→ query.py     (interactive queries)
                       └─→ validate.py  (regression chains)
```

## The five scripts

| Script | Runs | Reads | Writes | Time (full repo) |
|---|---|---|---|---|
| `cpp_index_driver.py`  | container | compile_commands.json | shards in `cache/tu_shards/` | ~30 min (cold) / seconds (warm cache) |
| `py_index.py`          | container | `.py` files | `cache/py_index.json` | ~3-5 s |
| `cpp_index_merger.py`  | container | shards | `cache/cpp_index/*.jsonl` | ~1-2 min |
| `stitch_sqlite.py`     | container | merged JSONL + py_index.json | `out/dep-graph.sqlite` | ~5-15 s |
| `validate.py`          | container | the SQLite | stdout | <1 s |

All container-side scripts run cleanly outside the container too IF you have
clang-20 + libclang Python bindings + pyyaml installed, and you supply paths
that match the host filesystem (no `/workspace` prefix). Inside the container
is the documented, supported path.

## Useful CLI flags

### `cpp_index_driver.py`
```
--workers N            number of parallel processes (0 = os.cpu_count())
--limit N              parse only first N to-do TUs (debugging)
--force                ignore cache; reparse everything in scope
--tu-list-file PATH    restrict to TUs listed in PATH (one absolute path/line)
--shard-root PATH      where to write per-TU shards (default cache/tu_shards)
```

### `cpp_index_merger.py`
```
--shard-root PATH      where to read shards from
--out PATH             output directory for merged JSONL
--tu-list-file PATH    include only shards whose TU is in the list (subset merging)
```

### `stitch_sqlite.py`
```
--cpp-dir PATH         merged C++ index dir
--py PATH              py_index.json
--out PATH             output .sqlite path
```

### `query.py` (subcommands; see `--help`)
```
callers SYMBOL          who calls this?
callees SYMBOL          what does this call?
blast SYMBOL --depth N  BFS upstream + downstream
by-file PATH            every node defined in a file
crosses SYMBOL          cross-language edges touching the symbol
bind PY_NAME            C++ targets bound to a Python name
find PATTERN            grep nodes by name/qname/file
```
SYMBOL may be either a node id (`cpp:c:@N@...`) or a unique qualified name
(`ttnn::add`). `--json` for machine output. `--db PATH` to point at a
non-default SQLite.

### `validate.py`
```
--db PATH              SQLite to test (default out/dep-graph.sqlite)
--chains PATH          YAML file with chains (default tests/expected_chains.yaml)
-q / --quiet           only print failures + summary
```

## Output / cache layout

```
dep-graph/
  scripts/         ← code; safe to read, not safe to edit casually
  tests/           ← validation chains
  cache/           ← regenerable intermediates (gitignored)
    tu_shards/      per-TU JSONL shards + manifests
    cpp_index/      merged JSONL for the whole C++ side
    py_index.json   Python AST output
  out/             ← final artifacts (gitignored)
    dep-graph.sqlite       full-repo DB
    dep-graph-demo.sqlite  small demo DB (~20 TU subset)
```

`cache/` and `out/` are safe to delete — they regenerate from sources.
`scripts/` and `tests/` are the only things under `dep-graph/` worth backing up.

## Smaller / partial runs

### "Quick demo build" (~30 sec wall, ~8 MB sqlite)

```bash
sudo docker exec --user 1000:1000 tt-metal-basic-dev-container bash -c '
  source /opt/venv/bin/activate && cd /workspace &&
  python dep-graph/scripts/cpp_index_driver.py \
      --db build_Release/compile_commands.json \
      --workers 14 \
      --tu-list-file dep-graph/cache/demo_tu_list.txt &&
  python dep-graph/scripts/cpp_index_merger.py \
      --tu-list-file dep-graph/cache/demo_tu_list.txt \
      --out dep-graph/cache/cpp_index_demo &&
  python dep-graph/scripts/stitch_sqlite.py \
      --cpp-dir dep-graph/cache/cpp_index_demo \
      --out dep-graph/out/dep-graph-demo.sqlite &&
  python dep-graph/scripts/validate.py \
      --db dep-graph/out/dep-graph-demo.sqlite
'
```

`dep-graph/cache/demo_tu_list.txt` is checked in; edit it to add/remove TUs.

### "Re-stitch only" (Python or C++ side updated, no re-parse)

```bash
sudo docker exec --user 1000:1000 tt-metal-basic-dev-container bash -c '
  source /opt/venv/bin/activate &&
  python /workspace/dep-graph/scripts/stitch_sqlite.py
'
```
Useful after you tweak `py_index.py` and re-run *only* it — the cpp cache
doesn't need to be rebuilt.

## Incremental behaviour

`cpp_index_driver.py` is mtime-aware:
- Each shard's `manifest.json` records the TU file's `mtime_ns` plus every
  included header's `mtime_ns` at parse time.
- On the next run, a shard is treated as fresh only if every recorded mtime
  still matches. Any header edit invalidates the dependent shard.
- Use `--force` to ignore the cache entirely.

Edit a single C++ file → next driver run reparses that TU plus any TU whose
manifest includes a header you touched. Everything else hits the cache.

## Troubleshooting

### libclang `ValueError: Unknown template argument kind 155`
Happens when clang-20 emits cursor kinds the Python bindings (libclang
18.1.1) don't recognize. `_cpp_lib.py`'s `_safe_kind` swallows these and
logs a diagnostic; if you see hundreds of failures here, the guard isn't
firing — check that `_walk` is calling `_safe_kind` rather than accessing
`cursor.kind` directly.

### "X TUs not in compile_commands.json"
Either the file path is wrong, or CMake didn't emit a command for it (header
files don't get their own TU entry; you index the `.cpp` that includes them).
Confirm with `grep '"file":' /workspace/build_Release/compile_commands.json`.

### PCH-related parse failures
Compile commands embed clang-20 precompiled-header args that the libclang
binding can't consume. `prune_args` in `_cpp_lib.py` strips them. If you see
fresh PCH errors, double-check that scrubbing still matches the actual argv
shape — it's: `-Xclang -include-pch -Xclang <pch> -Xclang -include -Xclang <hxx>`.

### Files owned by root showing up in the repo
You ran a `docker exec` without `--user 1000:1000`. Fix:
```bash
sudo chown -R ubuntu:ubuntu /home/ubuntu/tt-metal/dep-graph
```
Then make a habit of passing `--user 1000:1000` to every exec.

### `query.py` returns 0 callers/callees for a symbol you can see in `find`
Almost certainly the TU containing the *definition* (where edges originate
from) isn't in the indexed set yet. `find` matches header declarations too.
Try `query.py find <name>` and look for entries with `is_definition=1` —
they live in `.cpp`, not `.hpp`. If none of those exist, that TU hasn't been
indexed.

### Validation chain fails after editing a script
Don't panic — fix the chain or fix the script. The chains in
`dep-graph/tests/expected_chains.yaml` are the spec for "the graph still
behaves the way we expect." If you changed semantics on purpose, update the
chain to match.

## Scope (what's indexed, what isn't)

In scope (per opus-instructions.md §1):
- `tt_metal/` host code, including `tt_metal/impl/` and `tt_metal/jit_build/`
- `ttnn/cpp/` host code
- `tt_stl/`
- All Python under `ttnn/ttnn/`

Excluded:
- `tt_metal/hw/` (kernel firmware)
- `tt_metal/third_party/tt_llk_*/` (LLK)
- `runtime/sfpi/` (SFPI compiler)
- `.cpmcache/`, build dirs, `tests/`, `.github/`

To change: edit `DEFAULT_IN_SCOPE_PREFIXES` / `DEFAULT_OUT_OF_SCOPE_PREFIXES`
at the top of `dep-graph/scripts/_cpp_lib.py`, then `--force` the driver
(or wipe `cache/tu_shards/` and rerun).

## Schema crib sheet

`dep-graph.sqlite` tables:

| Table | Key columns |
|---|---|
| `nodes`            | `id` (PK), `language`, `kind`, `name`, `qualified_name`, `file`, `line_start/end`, `signature`, `is_definition`, `is_binding_target`, `is_binding_caller`, `attrs_json` |
| `edges`            | `src`, `dst`, `kind` (`calls`/`binds`/...), `site_file`, `site_line`, `crosses_language`, `via_decorator`, `via_helper` |
| `bindings`         | `python_name`, `cpp_node_id`, `cpp_qualified_name`, `helper` |
| `py_registrations` | `python_name`, `impl_node_id`, `impl_chain` (JSON), `decorator_label` |
| `meta`             | `schema_version`, `built_at` |

Useful indexes already in place: `nodes(name)`, `nodes(qualified_name)`,
`edges(src)`, `edges(dst)`, `edges(kind)`, `bindings(python_name)`,
`py_registrations(python_name)`.
