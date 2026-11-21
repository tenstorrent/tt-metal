# tt-triage

`tt-triage` is a tool that runs a series of checks/scripts to either diagnose a failure or provide insight into state of the system... It consists of the main program and a set of discoverable, user-defined scripts (python files)...

Triage is many small, readable scripts on top of a small framework, with a clean provider API and data that other repos can plug into.

# Running tt-triage

You can run `tt-triage` by executing `tools/tt-triage.py`.

- **Live:** attach while the process hangs, from another terminal.
- **Post-mortem:** CI's `TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE` (`.github/actions/setup-job/action.yml`) runs triage on a dispatch timeout, with `--sqlite-output-path` and `--llm-output-path` among its flags. Metal serializes Inspector first. CI runs triage once per job, on its first dispatch timeout, and every script runs, so a script that is slow or produces a lot of data costs every CI job that hangs.
- **Multi-host:** `tools/tt-run-triage.py` runs it per rank. Rank attribution lives there; scripts and serializers stay rank-agnostic.
- **Which devices:** `--dev` defaults to `in_use`, the devices the hung process used, according to Inspector or `TT_METAL_VISIBLE_DEVICES`. `--dev=all` covers every device tt-exalens sees, and `--dev=<id>` (repeatable) picks devices. `--loc` narrows the run to logical locations.

# Script discovery

`tt-triage` will search for scripts in the `tools/triage` directory. It will attempt to load all scripts and look for the following signature:
1. The script must define a global variable `script_config` of type `ScriptConfig`.
2. The script must define a `run` method with two arguments: `(args, context)`. `args` contains the parsed arguments for all scripts. `context` is a ttexalens `Context` object.

Discovery imports every `.py` file in the searched directories and skips any that doesn't load as a triage script, such as a helper module or a script whose import fails. In a full run the reason shows only at `--verbosity=4` or higher; `--run=<script>` reports the error.

## Scripts outside tools/triage

`--additional-scripts-directory=<dir>` adds `<dir>` to the search, alongside `tools/triage`. Repeat the option to add several directories. This lets a consuming repository keep its own triage scripts in its own tree while still using the built-in ones:

```bash
./tools/tt-triage.py --additional-scripts-directory=/path/to/my/custom/triage/scripts
```

Scripts in those directories are discovered, executed and given their arguments exactly like the built-in ones, and they may depend on built-in scripts by name. Each directory is also put on `sys.path`, so a script can import its siblings.

The option is read before discovery on every entry point, so it works the same way when a single script is run directly (see [Enabling standalone script execution](#enabling-standalone-script-execution)):

```bash
python /path/to/my/custom/triage/scripts/my_check.py --additional-scripts-directory=/path/to/my/other/triage/scripts
```

A script run directly does not need the flag for its own directory - that is always searched. A script and the scripts it depends on can sit side by side and be run with no options at all; the flag is only needed to reach scripts in *other* directories.

One limitation with `tt-run-triage`: its host-side argument check only knows the built-in scripts, so if a script in an additional directory defines its own options, passing them through `tt-run-triage` is rejected before the ranks start. Run `triage.py` under `tt-run` directly in that case.

**File names must be unique across all searched directories.** Scripts and their helper modules are imported by bare module name, so if two directories both contain `device_info.py`, only one of them can ever be imported under that name. `tt-triage` warns about the repeated names at startup and skips the shadowed script rather than running the wrong file under its path. That warning is the signal to watch for: `--run=<name>` resolves to the first match, which is the built-in one, so it runs the built-in script without complaining. Pointing at the shadowed file by path does report which file the name really resolved to. This applies to helper modules too - a `utils.py` next to your scripts would be shadowed by `tools/triage/utils.py`. `__init__.py` is exempt, since it is never loaded as a script.

# Design

## How it runs

`tools/tt-triage.py` is a thin entry point to `tools/triage/triage.py`, which does three things: it finds the scripts, runs them in dependency order, and writes out what they return.

**Finding the scripts.** Triage discovers every script in `tools/triage`, plus those in any `--additional-scripts-directory` (see [Script discovery](#script-discovery)).

**Running them.** Scripts are ordered by their `depends` lists, so a script runs after everything it depends on. Among scripts that are ready at the same time, a higher `priority` runs first (`ScriptPriority.HIGH`, `MEDIUM`, `LOW`), and the script's path breaks ties. Depending on a disabled script is an error that stops the whole run.

A run can start in two ways, and they take different code paths:

| Started as | What runs |
|---|---|
| `tt-triage.py`, alone or with `--run=all` | `main()` runs every discovered script, in the order above |
| `tt-triage.py --run=<script>`, or the script file itself | `run_script` runs that script and its dependencies only |

A rule about whether or how a script runs has to hold on both paths. `disabled` currently doesn't: see [Script configuration](#script-configuration).

**Writing the results.** Each script's `run(args, context)` returns data, and triage hands it to every output that was asked for:

| Output | Asked for with | Holds |
|---|---|---|
| Rich tables on the console | the default | the fields the verbosity level shows |
| CSV | `--llm-output` (on the console, in place of Rich) or `--llm-output-path` (to a file) | the same fields |
| SQLite, one table per script | `--sqlite-output-path` | every field, whatever `-v` says |

Failed checks and warnings go to a diagnostics section, or table, of each output.

A script's rows reach the outputs once, after its `run()` returns, so triage holds all of a script's rows in memory until then. Each output first turns every field it shows into text. The SQLite writer then stores a number as its original value, `None` as NULL and anything else as that text (see [Output columns](#output-columns)). In the code this is `init_output_serializer` and `serialize_result` in `triage.py`, and `MultiSerializer` and `extract_table_data` in `serializers.py`.

A few lines don't go through the outputs. `main()` prints them itself: a script skipped because a dependency failed, a data provider's failure, and the `--print-script-times` timings. `--triage-summary-path` writes one line per script, "pass" or "FAIL" with the reason; a data provider gets a line only when it fails. CI's `.github/scripts/utils/hang_report.py` puts that summary into the hung test's JUnit report.

**Where the data comes from.** Device state comes from tt-exalens (`Context`, `OnChipCoordinate`, `RiscDebug`, `ttexalens.tt_exalens_lib`). Host state, meaning what Metal created, comes from Inspector (see [Inspector and device ids](#inspector-and-device-ids)).

## Two kinds of scripts

- **Data provider**: `ScriptConfig(data_provider=True, depends=[...])`.
  - It reads data for other scripts; it never renders.
  - `run(args, context)` returns a data object, usually a class that other scripts query.
  - If the data object is slow to generate (such as when parsing long log files), use the `@triage_singleton` decorator on `run` to make it a singleton: it is built once and every script gets the same object. If the data being processed is not changing, it is up to the developer to cache it if needed.
  - Consumers import it as `from x_provider import run as get_x_provider, XProvider, XData`.
  - If a data provider fails, every script that depends on it fails too.
- **View / check script**: `ScriptConfig(depends=[...])`. It is thin: it gets providers, iterates with run_checks, and returns rows and/or logs checks. A check script can only check state (`check_noc_locations`), or check and return data (`check_arc`).
- Several views can share one provider. For example, `operation_provider` makes one pass over the cores for `dump_running_operations`, `dump_op_window` and `dump_op_mesh`.

### Provider shape

`semaphore_provider.py` and `circular_buffer_provider.py` are two examples of a provider. In outline:
```python
@dataclass
class SemaphoreData:
    kind: str = triage_field("Kind")  # program or global
    address: int = triage_field("Address", hex_serializer)
    value: int = triage_field("Value")

class SemaphoreProvider:
    def __init__(self, dispatcher_data, inspector_data, id_mapping, run_checks):
        ...  # every Inspector call happens here, once; build lookups

    @cache
    def get_cached_semaphores(self, location: OnChipCoordinate) -> list[SemaphoreData]:
        return self.get_semaphores(location)

    def get_semaphores(self, location: OnChipCoordinate) -> list[SemaphoreData]:
        ...  # read the device for this location

@triage_singleton
def run(args, context: Context) -> SemaphoreProvider:
    return SemaphoreProvider(get_dispatcher_data(args, context), ...)
```
The points below describe how the existing providers are written. Triage doesn't enforce them.
- When not: a check whose data no other script reads stays one script. None of the 10 `check_*` scripts is a provider; `check_noc_status` depends only on run_checks, dispatcher_data and elfs_cache. Data that public tt-exalens functions already return doesn't need a provider either: other scripts and repos can call those functions, so a view calls them directly.
- Use one provider per data domain. Getters are keyed by what they read (`location`, `risc_location` or `device`). Pair each `get_x` with a `@cache` `get_cached_x`, because several views ask for the same location in one run.
- The constructor makes every Inspector call up front, because each one is an RPC round trip. Per-location methods only read the device.
- Data classes carry `triage_field` metadata, so a consumer gets columns for free.
  - Compose with `recurse_field()`: `CallstacksData` embeds `DispatcherCoreData`.
  - Fields without `triage_field` are carried for code and never shown, e.g. `DispatcherCoreData.mailboxes`.
- Opt-in data gets a subclass (`CircularBufferDataWithContent(CircularBufferData)`), returned for every row when its read flag is set. Never gate it behind -vv; verbosity is the view's concern.
- Which script owns a flag:
  - A flag that changes what is read goes on the provider (`--include-cb-content`).
  - A flag that only changes what is shown goes in the view (`--include-global-unchanged`, `--all-cores`).
  - Provider docs never mention -v/-vv.

### View shape

A view is usually a few lines on top of a provider. `dump_semaphores.py` on `semaphore_provider` is one example, shown below. `dump_callstacks.py` and `dump_lightweight_asserts.py`, which both use `callstack_provider`, are another.
```python
script_config = ScriptConfig(depends=["run_checks", "semaphore_provider"])

def run(args, context: Context):
    include_unchanged: bool = args["--include-global-unchanged"]
    run_checks = get_run_checks(args, context)
    provider = get_semaphore_provider(args, context)
    return run_checks.run_per_block_check(
        lambda location: shown_semaphores(location, provider, include_unchanged),
        block_filter=["tensix", "active_eth", "idle_eth"],
    )

if __name__ == "__main__":
    run_script()
```

## run_checks (`run_checks.py`)

- It is the one way to iterate devices, blocks and cores, and it respects `--dev` and `--loc`:
  - `run_per_device_check(fn(device))`
  - `run_per_block_check(fn(location), block_filter=...)`
  - `run_per_core_check(fn(risc_debug), block_filter=..., core_filter=...)`
- `fn` returns `None` (no row), one dataclass (one row) or a list (several rows). The Dev / Loc / RiscV / NEO columns are prepended for you.
- It handles errors so check functions don't have to:
  - `TimeoutDeviceRegisterError` marks the device broken (and its remote devices) and skips it from then on. The rows from the blocks that finished before the timeout are kept; in `run_per_core_check`, the rows from earlier cores of the block that timed out are not.
  - `RiscHaltError` marks the core broken.
  - Any other exception becomes a "Skipping: ..." warning for that item.
- So don't wrap reads in try/except. If you must catch something to add context, `except Exception` is enough. `TimeoutDeviceRegisterError` derives from `HardwareError` → `BaseException` (tt-exalens 0.4.1), so it still reaches run_checks; older scripts' `except TimeoutDeviceRegisterError: raise` is redundant.
- Block types: `tensix`, `idle_eth`, `active_eth`, `eth`, `dram`. tt-exalens has more block types (ARC, PCIe, router-only and others), which run_checks doesn't iterate.
  - `eth` is every Ethernet block tt-exalens lists.
  - `idle_eth` and `active_eth` come from Inspector when it covers the device, otherwise from tt-exalens. From Inspector, a local Wormhole chip's list leaves out e0,15, which is reserved for syseng.
- Core types: `brisc`, `trisc0-3`, `ncrisc`, `erisc`/`erisc0`/`erisc1`, `drisc`, ...
- Typical `run()`: declare `BLOCK_TYPES_TO_CHECK` / `RISC_CORES_TO_CHECK` constants. Loop `for noc_id in (NocId.NOC0, NocId.NOC1)` with lambda default capture (`_noc_id=noc_id`).
- Broken and halted state lives in `triage_session.get_triage_session()`. `check_broken_components` (LOW priority) summarizes it.

## Scripts that change device state

- A normal triage run can leave cores halted. These reads through a RISC's debug interface halt the core: `get_pc()` when the core has no debug-bus PC signal, `--full-callstack` and `--gdb-callstack`, and ELF variable reads through `create_memory_access(risc_debug)` (`check_noc_status`).
- Continuing a core after a halt and a read or write can break it on Wormhole and Blackhole (tt-exalens#908). So `_patch_risc_debug` (`triage.py`) turns `cont()` and `continue_without_debug()` into no-ops on affected devices, and `ensure_halted()` only halts.
- The patched `halt()` records each core it halts in the triage session and does nothing for a core already recorded. A recorded core can still stop reporting halted on Blackhole. `check_broken_components` reports those cores at the end. So `halt()` returning doesn't mean the core is halted; check `is_halted()`.
- So a script that wants to see cores in the state the hang left them must run before anything that halts. A script that halts cores or otherwise changes the device should run after every other script: give it `ScriptPriority.LOW`. Priority only orders scripts that are ready to run, so a LOW script runs after every MEDIUM and HIGH script except one that depends on a LOW script, as `dump_risc_debug_signals` depends on `check_broken_components`. Among LOW scripts the path decides, so check where it lands against `check_arc` and `check_broken_components`. Say in its Description what it changes and that it runs last.
- Ethernet cores carry the traffic to remote chips, so debug access to them needs care: halting one can cut off the chip behind it.

## Checks and messages

- `log_check(ok, msg)` records a failure only when `ok` is False. It is not an info log: `log_check(True, ...)` does nothing. Failures are logged after the script has finished.
- `log_warning` (and its variants) records something worth noting that is not a failure. Use the variants that carry context instead of formatting it into the message: `log_check_device`, `log_check_location`, `log_check_risc`, and `log_warning*`.
- Log inline, where the check happens (`check_noc_locations`, `check_l1_status`). Don't collect findings to log later.
- Messages use Rich markup: `[error]...[/]`, `[info]...[/]`, `[warning]...[/]`. Console chatter that isn't a check goes through `utils.DEBUG/INFO/WARN`.
- Raise `TTTriageError` only when the script cannot continue and dependents must not run (missing ELF, no Inspector).

## Inspector and device ids

Inspector is the part of Metal's host runtime that keeps track of what the host created: programs and their kernels, mesh devices and workloads, build environments, sockets, global semaphores and so on (see `docs/source/tt-metalium/tools/inspector.rst`). Triage reads the device through tt-exalens, but the device alone doesn't say which program a core is running or where that program's ELF files are. Inspector does, and that is what lets triage turn the addresses and ids it reads from a chip into kernel names, callstacks and operations.

Triage gets Inspector's data in one of two ways, tried in this order:

1. Live, over RPC, while the process is still running. The port is 50051 plus the MPI rank.
2. From the files Metal serializes while it runs (`<method>.capnp.bin`), when the process is gone. Only methods without parameters are serialized. Triage looks in `--inspector-log-path`, else `$TT_METAL_LOGS_PATH/generated/inspector`, else `<tmp>/tt-metal/inspector`. Metal writes the files to `$TT_METAL_LOGS_PATH/generated/inspector`, which defaults to the current directory, so set the same `TT_METAL_LOGS_PATH` for both, or pass `--inspector-log-path`.

With neither, the `inspector_data` script raises `InspectorException`, and the scripts that depend on it don't run.

A script uses Inspector through the `inspector_data` provider: `inspector_data = get_inspector_data(args, context)`, then it calls the RPC methods on that object. The methods are defined in `tt_metal/impl/debug/inspector/rpc.capnp` (`tools/triage/inspector.capnp` is a symlink to it). Triage and Metal always come from the same build, so a script doesn't need a fallback for an older schema.

Three things to know when using it:

- Inspector speaks Metal device ids, which are not tt-exalens's ids under `TT_METAL_VISIBLE_DEVICES`. Translate with `metal_device_id_mapping` (`get_unique_id`, `has_metal_device_id`) and `run_checks.get_device_by_unique_id`.
- Most scripts don't need to call Inspector themselves. `dispatcher_data.get_cached_core_data(risc_location)` already combines it with what the core reports: the kernel, go message, launch slot, mailboxes, `kernel_config_base`, `programmable_core_type` and `host_assigned_id`. `dispatcher_data.risc_enabled(risc_name)` and `is_idle_in_default_view(risc_location)` say what the default view hides. Architecture differences live in `DispatcherData1xx` and its Wormhole and Blackhole subclasses.
- If a script needs host data that Inspector doesn't have, add it to Inspector (the page above shows how). Don't hardcode or guess it in triage: hardcoded values drift when Metal changes.

## Output

`tt-triage` visualizes the data scripts return, so describe the data accordingly. For example, `dump_callstacks` produces a table with all callstacks.

Verbosity is a *display* setting: it governs the console and `--llm-output` reports only. A database written with `--sqlite-output-path` always contains every column, whatever `-v` the run used, so the detail a `-vv` column would have shown can be queried afterwards without re-running triage.

To enable rich visualization, a script should return data as a tagged `@dataclass` or a list of tagged `@dataclass` objects of the same type. You control how each field is serialized and shown with these tagging methods:
- `triage_field(serialized_name, serializer, verbose=0)` – The field will be serialized (and visualized) as `serialized_name` (or the original field name if not provided) using the specified `serializer` (or `default_serializer`). The `verbose` parameter controls at which verbosity level the field is shown (0=always, 1=with `-v`, 2=with `-vv`). This controls how the field appears in the visualization; SQLite output stores the field regardless.
- `recurse_field(verbose=0)` – This will cause expansion of a field that is tagged as a `@dataclass`, so its internal fields are visualized as part of the parent. The `verbose` parameter controls the minimum verbosity level for this recursion.

Example:

```python
from dataclasses import dataclass
from triage import triage_field, recurse_field

@dataclass
class Chip:
    id: int = triage_field("Id")
    arch: str = triage_field("Arch")

@dataclass
class ChipCheck:
    # Visualized as "Check"
    check: str = triage_field("Check")
    # Expanded: its fields are visualized as part of ChipCheck
    chip: Chip = recurse_field()

def run(args, context):
    return [
        ChipCheck(check="Power OK", chip=Chip(id=1, arch="A0")),
        ChipCheck(check="Temp High", chip=Chip(id=2, arch="B1")),
    ]
```

The output looks like:
```
╭───────────┬────┬──────╮
│  Check    │ Id │ Arch │
├───────────┼────┼──────┤
│ Power OK  │  1 │  A0  │
│ Temp High │  2 │  B1  │
╰───────────┴────┴──────╯
```

### Output columns

- One row per entity ("one row per CB of the current program, per core"), stated in the Description.
- Column names are short Title Case, with units where they matter: `Speed MT/s`, `Size (pages)`, `Heartbeats/s`.
- One strictly typed value per column. The SQLite writer (`sqlite_serializer.py`) picks each column's type from the values in it: INTEGER or REAL when every value is numeric, TEXT otherwise. A value it can't store as a number is stored as its serialized string, which makes the whole column TEXT. That includes an int outside SQLite's signed 64-bit range.
  - No `"a/b/c"`, `"a -> b"` or space-joined strings.
  - Split into int columns, or use a categorical string (`Up`/`Down`).
  - Hex goes through `hex_serializer`; `None` renders as `N/A`.
- Detail columns of data that is always read get `verbose=1/2`. SQLite keeps them all anyway.
- Dump raw values (ints, hex, indices) rather than decoding when the decode adds little. Cover general state for any op. Consumer-specific naming belongs in a post-triage script in the consumer repo.
- A display tweak changes rendering only (`to_row` or the view), never aggregation keys, filters or row count.

### dump_callstacks output modes

By default, `dump_callstacks`:
- Shows only essential fields: Kernel ID:Name, Go Message, Subdevice, Preload, Waypoint, PC, and Callstack
- **Automatically filters out DONE cores** to reduce noise

You can control the output with two independent flags:

**Show more columns** using `-v` (can be repeated):
```bash
# Level 1: Add detailed fields (Firmware/Kernel Path, Host Assigned ID, Kernel Offset)
./tools/tt-triage.py --run=dump_callstacks -v

# Level 2: Add internal debug fields (RD PTR, Base, Offset)
./tools/tt-triage.py --run=dump_callstacks -vv
```

**Show all cores** (including DONE cores):
```bash
./tools/tt-triage.py --run=dump_callstacks --all-cores
```

**Combine both** for full details:
```bash
./tools/tt-triage.py --run=dump_callstacks --all-cores -vv
```

## Docstring

- The module docstring is docopt and is the user documentation:
  - `Usage:` (the script name and its flags)
  - `Options:` (only if there are flags)
  - `Description:`: what one row is, where each value comes from, units, caveats, and what fails the check
  - `Owner:` (a GitHub handle)
- Explain semantics there, not in code comments. `check_dram_health` and `dump_semaphores` are good models.
- **Docopt trap:** docopt parses the whole docstring, so a prose line that starts with `-` or `--flag` redefines the option and breaks every triage run. Keep the flag mid-line, and check that `grep -nE '^\s+-' script.py` shows only real Options lines.


# Writing a script

## Script configuration

Here is how the `ScriptConfig` class is defined:
```
@dataclass
class ScriptConfig:
    data_provider: bool = False
    disabled: bool = False
    depends: list[str] = []
    priority: ScriptPriority = ScriptPriority.MEDIUM
```
If you set `data_provider` to `True`, your script is treated as a data provider script.

If you set `disabled` to `True`, a full run skips your script. No enabled script may depend on it: triage checks the whole dependency graph before it runs anything, so one such dependency stops every run started through `tt-triage.py`, with or without `--run`. `--run=<script>` and running the script file directly still run a disabled script. To keep a script out of full runs unless asked for, derive `disabled` from an environment variable, as `dump_aggregated_callstacks` does.

`depends` is a list of scripts that your script depends on. Scripts that iterate devices, blocks or cores depend on `run_checks`. Scripts that depend on checking operations, kernels, or firmware depend on the `dispatcher_data` script.

A bare dependency name is looked for next to your script first, then in `tools/triage`, then in the directories given by `--additional-scripts-directory`, in the order they were given; the first match wins. An absolute path is used as given. This is what lets a script outside `tools/triage` depend on a built-in provider such as `run_checks` by name.

`priority` orders scripts that are ready to run at the same time: HIGH first, LOW last (see [How it runs](#how-it-runs) and [Scripts that change device state](#scripts-that-change-device-state)).

## Additional pip requirements

If your `tt-triage` script requires extra Python libraries, add the package name (and version if needed) to `tools/triage/requirements.txt`.
`triage.py` exits at import with an install hint if a required module is missing.

## Enabling standalone script execution

If you would like to enable your script to be executed as a standalone script, add these lines to the end:
```python
if __name__ == "__main__":
    from triage import run_script

    run_script()
```
Or, if you import `run_script` where you import `ScriptConfig`, it would look like this:
```python
from triage import ScriptConfig, run_script

script_config = ScriptConfig(...)

...

if __name__ == "__main__":
    run_script()
```

If you don't want to execute your script directly, you can still do so by running: `tools/tt-triage.py --run=your_script`. You can append more `--run=<selected_script>` arguments to run multiple

# Development

## Using a tt-exalens checkout

- Triage imports tt-exalens as a library. `tools/triage/requirements.txt` pins its version, and the Python environment that runs triage usually has that wheel installed.
- A change that needs a tt-exalens change can't be tested against the pinned wheel. Run triage with `PYTHONPATH=<tt-exalens checkout>` from a Python environment that has triage's requirements (`tools/triage/requirements.txt`: pycapnp and a tt-exalens wheel) and the checkout's requirements (`ttexalens/requirements.txt`, which may pin a newer tt-umd than the wheel). The checkout must be built, because its native module comes from the build.
- The version check reads the installed wheel's metadata, not the checkout: a wheel must be installed, and `--skip-version-check` only skips a version mismatch.
- Shipping such a change takes a tt-exalens release and a pin bump in `tools/triage/requirements.txt`.

## Type checks and lint

- Run `pre-commit run --files <files>`. Its triage type-check hook runs `tools/triage/run_type_checks.sh` (mypy and basedpyright, configured by `mypy.ini` and `pyrightconfig.json` next to it).
- Unless `SKIP_INSTALL=1`, the script installs `requirements-dev.txt` (which includes the pinned tt-exalens and tt-umd) into the current environment when ttexalens, mypy or basedpyright is missing. It regenerates the untracked `inspector_capnp.pyi` from the Inspector schema on every run.
- To type-check against a tt-exalens checkout instead of the installed wheel: `SKIP_INSTALL=1 PYTHONPATH=<tt-exalens checkout> tools/triage/run_type_checks.sh`. Against a checkout newer than the pin, scripts you didn't change can fail where the API changed; run the same command on unmodified main and compare.

## Tests (`tools/tests/triage/`)

- `test_triage.py`, `class TestTriage`: a class-scoped `cause_hang_with_app` fixture starts one hang app per config:
  - `add_2_integers_hang` with fast dispatch and lightweight asserts;
  - `add_2_integers_hang` with slow dispatch;
  - `ttnn_add_integers_hang` with fast and slow dispatch.
- `TestMeshSocketTriage` uses `mesh_socket_hang`. Hang apps live in `tools/tests/triage/hang_apps/` (built into `build/tools/tests/triage/hang_apps/`). A hang app that is a Python script runs under the test's own interpreter (`sys.executable`), so the environment that runs the tests needs ttnn as well as triage's requirements.
- A test calls `result = self.run_triage_script("dump_x.py")` and asserts on row fields. The helper fails the test if any check failure was logged, and `test_triage_executes_no_errors` runs every script. So a new script must run clean on the hang apps. These tests don't pass `--sqlite-output-path`.
- While iterating, run one test on one config: `pytest tools/tests/triage/test_triage.py -k "test_x and <param id>"`. The full file takes minutes.
- To test new data, plant it in the hang app with known values and assert exact rows. In `test_dump_semaphores`, a program semaphore starts at 7 and is bumped twice to 9, and a global one is set to 3.
- No unit tests of decoders or behavior on synthetic data. A drift test against metal's headers is the only test wanted for hardcoded addresses, and stopgap code gets none.
- Other suites: `test_sqlite_serializer.py`, `test_additional_scripts_directory.py`, `test_hang_report.py`, `test_ttrun_rank.py`.

## Checking a change on a real hang

1. Start `build/tools/tests/triage/hang_apps/add_2_integers_hang/triage_hang_app_add_2_integers_hang` in the background with `TT_METAL_OPERATION_TIMEOUT_SECONDS=0.5 TT_TRIAGE_HANG_APP_STAY_ALIVE=1` and TT_METAL_LOGS_PATH unset, as the `test_triage.py` fixture does.
2. Wait until it prints "Device timeout detected as expected."; it then stays alive so triage can reach its Inspector RPC. With `TT_TRIAGE_HANG_APP_STAY_ALIVE` set, the app is killed when its parent process exits (`PR_SET_PDEATHSIG`), so start it from a parent that outlives the triage run.
3. Run `python tools/tt-triage.py --run=<script>`.
4. SIGKILL the app; it ignores SIGINT. Find it with `pgrep -x triage_hang_app`, not `pgrep -f`, which also matches your own shell.
5. Reset the device with `tt-smi -r`.

To compare triage output before and after a change, take two baselines on fresh hangs. The console differs between runs of the same code: callstack depth and halt warnings vary. Compare per script section, and compare the SQLite tables' schemas and row counts.
