# Create TMP Migration JIRA Ticket

**Purpose**: Generate a JIRA ticket description for migrating a device operation to the TMP (Template Metaprogramming) pattern.

## ⚠️ CRITICAL: User Confirmation Required

**NEVER create a JIRA ticket without explicit user confirmation.** Always:
1. Generate the ticket description first
2. Show the user the complete ticket details (summary, description, metadata)
3. **Wait for explicit confirmation** before creating the ticket in JIRA
4. Only proceed with ticket creation after the user explicitly approves

## Usage

When creating a new TMP migration ticket, provide:
1. **Operation name** (e.g., `transpose`, `argmax`, `embedding`, `prod`)
2. **Operation path** (e.g., `data_movement/transpose/device/transpose_op`, `reduction/prod/device/prod_nc_op`)
3. **GitHub issue number** (optional, e.g., `32761`, `32699`)
4. **Test directory/path** (optional) - Use {{@.cursor/commands/ttnn/find-operation-tests.md}} to identify the correct test paths and CI jobs during initial investigation. If not provided, the testing path will be marked as an assumption.

## JIRA Ticket Configuration

When creating the ticket, use the following settings:
- **Project**: TENSAICA (tt-metal)
- **Issue Type**: Task
- **Summary Format**: `[tt-metal] TMP migration: {{OPERATION_PATH}}`
- **Priority**: Major
- **Story Points**: 3
- **Assignee**: Current user (Ilia Shutov)
- **Sprint**: Add to the same sprint as similar tickets (may require manual assignment in JIRA board)

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
pytest tests/ttnn/unit_tests/operations/{{TEST_PATH}} -v
{code}

*Note*: The main unit test for {{OPERATION_NAME}} operation is located in {{tests/ttnn/unit_tests/operations/{{TEST_DIRECTORY}}/test_{{TEST_FILE}}.py}} (function {{test_{{OPERATION_NAME}}}}). Additional tests may exist in:
* {{tests/tt_eager/python_api_testing/unit_testing/test_{{OPERATION_NAME}}_*.py}}
* {{tests/sweep_framework/sweeps/{{CATEGORY}}/{{OPERATION_NAME}}.py}}

If no information about unit tests is provided, the testing pytest path above is only an assumption. Usually, this information (which tests and CI jobs will run the operation's validation) is obtained during initial investigation when using the {{@.cursor/commands/ttnn/find-operation-tests.md}} Cursor command.
```

## Example

For operation `prod` with path `reduction/prod/device/prod_nc_op` and GitHub issue `32699`:

1. **Summary**: `[tt-metal] TMP migration: reduction/prod/device/prod_nc_op`
2. Replace `{{OPERATION_NAME}}` with `prod`
3. Replace `{{OPERATION_PATH}}` with `reduction/prod/device/prod_nc_op`
4. Replace `{{GITHUB_ISSUE_NUMBER}}` with `32699` (in both places)
5. Replace `{{TEST_PATH}}` with `tests/ttnn/unit_tests/operations/reduce/test_reduction.py::test_prod`
6. Replace `{{TEST_DIRECTORY}}` with `reduce`
7. Replace `{{TEST_FILE}}` with `reduction`
8. Replace `{{CATEGORY}}` with `reduction`

## Instructions for AI Assistant

### Step 1: Gather Information

When the user requests to create a TMP migration ticket, ask for:
1. Operation name (e.g., "transpose", "argmax", "prod")
2. Operation path (e.g., "data_movement/transpose/device/transpose_op", "reduction/prod/device/prod_nc_op")
3. GitHub issue number (optional, e.g., "32761", "32699")
4. Test directory/path (optional) - if not provided, search the codebase to find test files and use the most relevant one

### Step 2: Search for Test Information

Before generating the ticket:
- Search for test files related to the operation (e.g., `test_*prod*.py`, `test_*transpose*.py`)
- Identify the main unit test file and function name
- Note additional test locations if found

### Step 3: Generate Ticket Details

Generate the complete ticket information including:
- **Summary**: `[tt-metal] TMP migration: {{OPERATION_PATH}}`
- **Description**: Replace all placeholders in the template
- **Metadata**: Priority (Major), Story Points (3), Assignee (current user)

### Step 4: ⚠️ CRITICAL - Request Confirmation

**DO NOT CREATE THE TICKET YET.** Instead:
1. Display the complete ticket details to the user:
   - Summary
   - Full description
   - All metadata (Priority, Story Points, Assignee, etc.)
2. Explicitly ask: "Please review the ticket details above. Should I create this ticket in JIRA?"
3. **Wait for explicit confirmation** before proceeding

### Step 5: Create Ticket (Only After Confirmation)

Only after receiving explicit user confirmation:
1. Create the JIRA ticket with:
   - Project: TENSAICA
   - Issue Type: Task
   - Summary: `[tt-metal] TMP migration: {{OPERATION_PATH}}`
   - Description: Generated description
   - Priority: Major
   - Story Points: 3
   - Assignee: Current user
2. Note that sprint assignment may need to be done manually in JIRA board
3. Provide the ticket key and URL to the user

**Important Notes**:
- If test information is not provided, the testing path in the generated description should be marked as an assumption
- Recommend using the {{@.cursor/commands/ttnn/find-operation-tests.md}} command during initial investigation to identify the correct test paths and CI jobs that validate the operation
- Always verify the operation path format matches existing tickets (e.g., `reduction/prod/device/prod_nc_op`)
