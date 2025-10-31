# tt-triage

`tt-triage` is a tool that runs a series of checks/scripts to either diagnose a failure or provide insight into state of the system... It consists of the main program and a set of discoverable, user-defined scripts (python files)...

# Running tt-triage

You can run `tt-triage` by executing `tools/tt-triage.py`.

# Script discovery

`tt-triage` will search for scripts in the `tools/triage` directory. It will attempt to load all scripts and look for the following signature:
1. The script must define a global variable `script_config` of type `ScriptConfig`.
2. The script must define a `run` method with two arguments: `(args, context)`. `args` contains the parsed arguments for all scripts. `context` is a ttexalens `Context` object.

There are two types of scripts:
- Data provider
- State checker

## Data provider scripts

Data provider scripts return data from their `run` method. `run` should return a data object, which is usually a class that can be queried for information.

If the data object is slow to generate (such as when parsing long log files), it should use the `@triage_singleton` decorator to make it a singleton. If the data being processed is not changing, it is up to the developer to cache their data if needed.

If a data provider script fails, all scripts that depend on it will also fail.

## State checker scripts

A state checker script can only check state (see `check_noc_locations` for an example), or it can perform checks and return data as well (see `check_arc` for an example).

To log a check failure but continue execution, use the `log_check` method. It will log failures after the script has finished executing.

If a check is critical (such as a missing ELF file), the script should raise a `TTTriageError` exception. Critical error means that script cannot advance without that check and that all dependent scripts shouldn't be executed.

### Data visualization

`tt-triage` will try to visualize data returned by scripts, so it is important to describe the data accordingly. For example, see `dump_callstacks` — `tt-triage` would generate a table with all callstacks.

#### dump_callstacks output modes

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

This keeps the default output clean while allowing detailed inspection when needed.

To enable rich visualization, a checker script should return data as a tagged `@dataclass` or a list of tagged `@dataclass` objects of the same type. Visualization in `tt-triage` is achieved by serializing data fields in a way that describes how they should appear in the output. You control this by using tagging methods and their arguments to specify how each field should be serialized and thus visualized:
- `triage_field(serialized_name, serializer, verbose=0)` – The field will be serialized (and visualized) as `serialized_name` (or the original field name if not provided) using the specified `serializer` (or `default_serializer`). The `verbose` parameter controls at which verbosity level the field is shown (0=always, 1=with `-v`, 2=with `-vv`). This controls how the field appears in the visualization.
- `recurse_field(verbose=0)` – This will cause expansion of a field that is tagged as a `@dataclass`, so its internal fields are visualized as part of the parent. The `verbose` parameter controls the minimum verbosity level for this recursion.
  ```python
  @dataclass
  class ChipCheck:
    # Regular field that will be visualized under the `Check` name
    check: str = triage_field("Check")
    # Field that will be expanded and visualized as its internal fields (see previous example)
    chip: Chip = recurse_field()
    # Advanced data only shown with -vv
    debug_data: DebugInfo = recurse_field(verbose=2)
  ```

By carefully specifying these serialization options, you control how your data will be visualized in the `tt-triage` output.

#### Example: Full Visualization

Below is a representative example showing how to define data classes for visualization, including all three field tagging methods:

```python
from dataclasses import dataclass
from triage import triage_field, recurse_field, collection_serializer

@dataclass
class Chip:
    # This field will be visualized as "ID:Arch" with value "id:arch"
    id: int = triage_field("Id")
    # This field is only used for combination above, not visualized separately
    arch: str = triage_field("Arch")

@dataclass
class ChipCheck:
    # This field will be visualized as "Check"
    check: str = triage_field("Check")
    # This field will be expanded and its fields visualized as part of ChipCheck
    chip: Chip = recurse_field()

# Example usage in a script's run() function:
def run(args, context):
    # Return a list of ChipCheck objects for visualization
    return [
        ChipCheck(check="Power OK", chip=Chip(id=1, arch="A0")),
        ChipCheck(check="Temp High", chip=Chip(id=2, arch="B1")),
    ]
```

When this data is returned from your script, `tt-triage` will visualize it as a table with columns "Check" and "ID:Arch", and rows for each `ChipCheck` instance.

For example, the output might look like:
```
╭───────────┬────┬──────╮
│  Check    │ Id │ Arch │
├───────────┼────┼──────┤
│ Power OK  │  1 │  A0  │
│ Temp High │  2 │  B1  │
╰───────────┴────┴──────╯
```

## Script configuration

Here is how the `ScriptConfig` class is defined:
```
@dataclass
class ScriptConfig:
    data_provider: bool = False
    disabled: bool = False
    depends: list[str] = []
```
If you set `data_provider` to `True`, your script is treated as a data provider script.

If you set `disabled` to `True`, your script will not be executed (and all scripts that depend on it will fail).

`depends` is a list of scripts that your script depends on. Scripts that perform checks per device depend on the `check_per_device` script. Scripts that depend on checking operations, kernels, or firmware depend on the `dispatcher_data` script.

## Additional pip requirements

If your `tt-triage` script requires extra Python libraries, add the package name (and version if needed) to `tools/triage/requirements.txt`.
After updating `requirements.txt`, also update `triage.py` in the section "Check if requirements are installed" to ensure your dependency is checked and installed automatically.

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
