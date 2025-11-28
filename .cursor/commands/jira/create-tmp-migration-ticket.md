# Create TMP Migration JIRA Ticket

**Purpose**: Generate a JIRA ticket description for migrating a device operation to the TMP (Template Metaprogramming) pattern.

## Usage

When creating a new TMP migration ticket, provide:
1. **Operation name** (e.g., `transpose`, `argmax`, `embedding`)
2. **Operation path** (e.g., `data_movement/transpose/device/transpose_op`, `reduction/argmax/device/argmax_op`)
3. **GitHub issue number** (optional, e.g., `32761`)
4. **Test directory/path** (optional) - Use {{@.cursor/commands/ttnn/find-operation-tests.md}} to identify the correct test paths and CI jobs during initial investigation. If not provided, the testing path will be marked as an assumption.

## Generated JIRA Description Template

```markdown
h2. Overview

Migrate the {{OPERATION_NAME}} device operation from the old vector-based structure to the new TMP (Template Metaprogramming) pattern to eliminate unnecessary heap allocations.

*Target Operation*: {{OPERATION_PATH}}
*Related Issue*: [GitHub Issue #GITHUB_ISSUE_NUMBER|https://github.com/tenstorrent/tt-metal/issues/GITHUB_ISSUE_NUMBER]

h2. Migration Guide

Follow the detailed migration guide: {{ttnn/cursor/DEVICE_OPERATION_MIGRATION_GUIDE.md}}

h2. Migration Checklist

* [ ] Step 1: Created {{operation_attributes_t}} struct with all const configuration members
* [ ] Step 2: Created {{tensor_args_t}} struct with all Tensor parameters from invoke signature
* [ ] Step 3: Defined {{tensor_return_value_t}} and {{spec_return_value_t}} appropriately
* [ ] Step 4: Implemented {{compute_output_specs}}
* [ ] Step 5: [Optional] Implemented {{create_output_tensors}} (if legacy had it)
* [ ] Step 6: Implemented {{select_program_factory}} returning correct variant type
* [ ] Step 6a: [If needed] Created separate mesh workload factory for {{mesh_coords}} filtering support
* [ ] Step 7: Implemented {{validate_on_program_cache_miss}}
* [ ] Step 8: [Optional] Implemented {{validate_on_program_cache_hit}} (if legacy had it)
* [ ] Step 9: Registered prim in {{ttnn::prim}} namespace
* [ ] Step 10: Updated all call sites to use prim instead of direct invoke or {{operation::run}}
* [ ] Step 11: Created program factory with:
  * [ ] {{shared_variables_t}} struct (from lambda captures)
  * [ ] {{create}} method (from old {{create_program}})
  * [ ] {{override_runtime_arguments}} method (from lambda body)
* [ ] Step 12: [Optional] Implemented {{compute_program_hash}} (if legacy had it)
* [ ] Step 13: Removed old device operation code (after verification)
* [ ] Step 14: Relevant tests pass
* [ ] Step 15: Code compiles without warnings

h2. Testing

{code:bash}
source python_env/bin/activate
pytest tests/ttnn/unit_tests/operations/{{OPERATION_NAME}}/
{code}

*Note*: If no information about unit tests is provided, the testing pytest path above is only an assumption. Usually, this information (which tests and CI jobs will run the operation's validation) is obtained during initial investigation when using the {{@.cursor/commands/ttnn/find-operation-tests.md}} Cursor command.
```

## Example

For operation `transpose` with path `data_movement/transpose/device/transpose_op` and GitHub issue `32761`:

1. Replace `{{OPERATION_NAME}}` with `transpose`
2. Replace `{{OPERATION_PATH}}` with `data_movement/transpose/device/transpose_op`
3. Replace `{{GITHUB_ISSUE_NUMBER}}` with `32761` (in both places)

## Instructions for AI Assistant

When the user requests to create a TMP migration ticket, ask for:
1. Operation name (e.g., "transpose", "argmax")
2. Operation path (e.g., "data_movement/transpose/device/transpose_op")
3. GitHub issue number (optional)
4. Test directory/path (optional) - if not provided, use assumption and note it

**Important**: If test information is not provided, the testing path in the generated description should be marked as an assumption. Recommend using the {{@.cursor/commands/ttnn/find-operation-tests.md}} command during initial investigation to identify the correct test paths and CI jobs that validate the operation.

Then generate the JIRA description by replacing the placeholders in the template above.
