# Running tt-triage

You can run `tt-triage` by executing script `scripts/debugging_scripts/tt-triage.py`.
If you are developing a script or you want to run a single script (e.g. `dump_callstacks.py`) you can just execute it with: `scripts/debugging_scripts/tt-triage.py --run=dump_callstacks`. You can append more `--run=<selected_script>` to run multiple scripts. If script has implemented main, you can just executed with: `scripts/debugging_scripts/dump_callstacks.py`.

# Script discovery

`tt-triage` will search for its scripts in `scripts/debugging_scripts` directory. It will try to load all scripts and search for its signature:
1. Script needs to define global variable `script_config` of type `ScriptConfig`
2. Script needs to define `run` method with two arguments: `(args, context)`. `args` is parsed arguments for all scripts. `context` is ttexalens `Context` object.

There are two types of scripts:
- Data provider
- State checker

## Data provider scripts

Data provider scripts return data from its `run` method. `run` should return data object, which is usually a class that can be queried for data.
If data object is slow to generate (like parsing long log files), it should use `@triage_cache` decorator to make it a singleton. Making it a singleton doesn't mean that data inside is cached. If data provider script fails, all scripts that depend on it will fail as well.

## State checker scripts

State checker script can only check state (look at `check_noc_locations`) or it can do checks and return data as well (look at `check_arc`).
To log check failure, but continue execution use `log_check` method. It will log failures after script has finished its execution.
If check is important (like missing elf file), script should raise `TTTriageError` exception.

### Data serialization

You should return data from your checker script as a tagged `@dataclass` or list of tagged `@dataclass` objects of same type.
There are couple of field tagging you can use:
- `triage_field(serialized_name, serializer)` - field will be serialized as `serialized_name` (or original field name if not provided) using specified `serializer` (or `default_serializer`)
- `combined_field(additional_fields, serialized_name, serializer)` - field will be serialized together with `additional_fields` under `serialized_name` using `serializer`. If none of the parameters is provided, serialization will be ignored as it will be serialized with different field. Example:
  ```python
  @dataclass
  class Chip:
    # this field will trigger serialization
    id: int = combined_field("arch", "ID:Arch", collection_serializer(":"))
    # this will be ignored by serialization
    arch: str = combined_field()
  ```
- `resurse_field()` - thiss will cause expanding of field that is tagged `@dataclass`
  ```python
  @dataclass
  class ChipCheck:
    # Regular field that will be serialized under `Check` name
    check: str = triage_field("Check")
    # Field that will be expanded and serialized as its internal fields (see previous example)
    chip: Chip = recurse_field()
  ```

## Script configuration

Here is how `ScriptConfig` class is defined:
```
@dataclass
class ScriptConfig:
    data_provider: bool = False
    disabled: bool = False
    depends: list[str] = []
```
If you set `data_provider` to `True` your script is treated as data provider script.

If you set `disabled` to `True` your script will not be executed (and all scripts that depend on it will fail).

`depends` is list of scripts that your script depend on. Scripts that do checks per device depend on script `check_per_device`. Scripts that depend on checking operations, kernels, firmware, depend on `dispatcher_data` script.

## Enabling script standalone execution

If you would like to enable script so it can be executed as standalone script, you should add these lines to the end:
```python
if __name__ == "__main__":
    from triage import run_script

    run_script()
```
or if you import `run_script` where you import `ScriptConfig` then it would look something like this:
```python
from triage import ScriptConfig, run_script

script_config = ScriptConfig(...)

...

if __name__ == "__main__":
    run_script()
```
