# Migrate Device Operation to TMP Pattern

Migrate a device operation from the old vector-based structure to the new TMP (Template Metaprogramming) device operation pattern that eliminates unnecessary heap allocations.

## Usage

When you need to migrate a device operation, use this command and provide:
- The operation name you're migrating (e.g., 'Embedding', 'Unary', 'Dropout')
- The location of the old device operation code

## Comprehensive Guide

**Follow the detailed migration guide at:** `ttnn/cursor/DEVICE_OPERATION_MIGRATION_GUIDE.md`

The comprehensive guide includes:
- Detailed comparison of old vs new operation structures
- Step-by-step migration process with code examples
- Complete 15-step checklist for verification
- File structure guidance
- Building and testing instructions
- Reference examples (Dropout operation and send_async operation)

See the comprehensive guide for detailed examples and the complete 15-step checklist.

## Common Pitfalls

1. **Forgetting to register the prim**: Always register in `ttnn::prim` namespace and use it instead of direct calls
2. **Including runtime-only values in hash**: Only hash compile-time constants that affect program structure
3. **Not including values that affect the program structure in hash**: Every parameter that affects program structure must be in the hash
4. **Redundant tensors in tensor_args_t**: Do not add redundant arguments like `preallocated_output`, if legacy operation did not handle that explicitly in `create_output_tensors`
5. **tensor_args_t must NOT contain references** - no `const &` or `&`

## Example Reference

See the Dropout operation migration for a complete example:
- Location: `ttnn/cpp/ttnn/operations/experimental/dropout`
- PRs: https://github.com/tenstorrent/tt-metal/pull/11793, https://github.com/tenstorrent/tt-metal/pull/11956

See send_async operation migration for a CCL example:
- Location: `ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/send_async`
- PR: https://github.com/tenstorrent/tt-metal/pull/33005

For detailed file structure, building, and testing instructions, see the comprehensive guide.
