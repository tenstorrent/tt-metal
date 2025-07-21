# Running tt-triage

You can run `tt-triage` by executing `scripts/debugging_scripts/tt-triage.py`.

# Script discovery

`tt-triage` will search for scripts in the `scripts/debugging_scripts` directory. It will attempt to load all scripts and look for the following signature:
1. The script must define a global variable `script_config` of type `ScriptConfig`.
2. The script must define a `run` method with two arguments: `(args, context)`. `args` contains the parsed arguments for all scripts. `context` is a ttexalens `Context` object.

There are two types of scripts:
- Data provider
- State checker

## Data provider scripts

Data provider scripts return data from their `run` method. `run` should return a data object, which is usually a class that can be queried for information.

If the data object is slow to generate (such as when parsing long log files), it should use the `@triage_cache` decorator to make it a singleton. Making it a singleton does not mean that the data inside is cached.

If a data provider script fails, all scripts that depend on it will also fail.

## State checker scripts

A state checker script can only check state (see `check_noc_locations` for an example), or it can perform checks and return data as well (see `check_arc` for an example).

To log a check failure but continue execution, use the `log_check` method. It will log failures after the script has finished executing.

If a check is critical (such as a missing ELF file), the script should raise a `TTTriageError` exception.

### Data serialization

`tt-triage` will try to visualize data returned by scripts, so it is important to describe the data accordingly. See `dump_callstacks` for an example — `tt-triage` would generate a table with all callstacks.

To describe data, a checker script should return data as a tagged `@dataclass` or a list of tagged `@dataclass` objects of the same type. Each field of that class should also describe its fields by initializing its value with different tagging methods:
- `triage_field(serialized_name, serializer)` – The field will be serialized as `serialized_name` (or the original field name if not provided) using the specified `serializer` (or `default_serializer`).
- `combined_field(additional_fields, serialized_name, serializer)` – The field will be serialized together with `additional_fields` under `serialized_name` using `serializer`. If none of the parameters are provided, serialization will be ignored, as it will be serialized with a different field. Example:
  ```python
  @dataclass
  class Chip:
    # This field will trigger serialization
    id: int = combined_field("arch", "ID:Arch", collection_serializer(":"))
    # This will be ignored by serialization
    arch: str = combined_field()
  ```
- `recurse_field()` – This will cause expansion of a field that is tagged as a `@dataclass`.
  ```python
  @dataclass
  class ChipCheck:
    # Regular field that will be serialized under the `Check` name
    check: str = triage_field("Check")
    # Field that will be expanded and serialized as its internal fields (see previous example)
    chip: Chip = recurse_field()
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

If you don't want to execute your script directly, you can still do so by running: `scripts/debugging_scripts/tt-triage.py --run=your_script`. You can append more `--run=<selected_script>` arguments to run multiple
