# Testing Block Variants

This document describes how to test the newly added block variants for the Compute API.

## New Block Variants Added

### Eltwise Binary Operations (`eltwise_binary.h`)
- `add_block_init<Ht, Wt>()` - Initialize block-level addition
- `add_block<Ht, Wt>()` - Perform block-level addition (L1 → DEST)
- `sub_block_init<Ht, Wt>()` - Initialize block-level subtraction
- `sub_block<Ht, Wt>()` - Perform block-level subtraction (L1 → DEST)
- `mul_block_init<Ht, Wt>()` - Initialize block-level multiplication
- `mul_block<Ht, Wt>()` - Perform block-level multiplication (L1 → DEST)

### Reduce Operations (`reduce_custom.h`)
- `reduce_block_init<reduce_type, reduce_dim, Ht, Wt>()` - Initialize block-level reduce
- `reduce_block<reduce_type, reduce_dim, Ht, Wt>()` - Perform block-level reduce (L1 → DEST)
- `reduce_block_uninit()` - Cleanup after block-level reduce

### Pack Operations (`pack.h`)
- `pack_block<Ht, Wt>()` - Pack block of tiles from DEST → L1
- `pack_reduce_block<reduce_dim, Ht, Wt>()` - Pack reduced block from DEST → L1

## Testing Approach

### Unit Test Pattern

Create a compute kernel that uses the block variants:

```cpp
// Example compute kernel using add_block
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/pack.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t Ht = 2;  // Block height
    constexpr uint32_t Wt = 4;  // Block width

    uint32_t cb_in0 = 0;
    uint32_t cb_in1 = 1;
    uint32_t cb_out = 16;

    // Initialize
    add_block_init<Ht, Wt>(cb_in0, cb_in1);

    // Acquire DEST
    acquire_dst();

    // Perform block operation
    add_block<Ht, Wt>(cb_in0, cb_in1, 0, 0, 0);

    // Pack result
    pack_block<Ht, Wt>(0, cb_out);

    // Release DEST
    release_dst();
}
}
```

### Python Test Pattern

```python
import pytest
import torch
import ttnn
from models.common.utility_functions import comp_pcc

@pytest.mark.parametrize("Ht,Wt", [(1,1), (2,2), (2,4), (4,4)])
def test_add_block(device, Ht, Wt):
    """Test add_block variant"""
    batch = 1
    h = Ht * 32  # tiles are 32x32
    w = Wt * 32

    # Create input tensors
    torch_a = torch.randn((batch, 1, h, w)).bfloat16()
    torch_b = torch.randn((batch, 1, h, w)).bfloat16()

    # Expected output
    torch_out = torch_a + torch_b

    # Convert to TT tensors
    tt_a = ttnn.from_torch(torch_a, device=device, layout=ttnn.TILE_LAYOUT)
    tt_b = ttnn.from_torch(torch_b, device=device, layout=ttnn.TILE_LAYOUT)

    # Run operation (would need custom kernel with add_block)
    # tt_out = custom_add_block_op(tt_a, tt_b, Ht, Wt)

    # Convert back and compare
    # output = ttnn.to_torch(tt_out)
    # passing, pcc = comp_pcc(torch_out, output, 0.9999)
    # assert passing
```

### Integration Test Checklist

For each new block variant:
- [ ] Test with minimum block size (1x1)
- [ ] Test with various block sizes (2x2, 2x4, 4x4, 4x8)
- [ ] Test with maximum DEST capacity (16 tiles total)
- [ ] Test on Blackhole architecture
- [ ] Test on Wormhole B0 architecture
- [ ] Verify PCC > 0.9999 against reference implementation
- [ ] Test with different data types (bfloat16, fp16, fp32 if supported)
- [ ] Verify DEST capacity static assertions trigger correctly

## Manual Verification

### Build
```bash
cd /localdev/ncvetkovic/reconfig/tt-metal
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
./build_metal.sh
```

### Run Existing Tests
```bash
source python_env/bin/activate
pytest tests/ttnn/unit_tests/operations/eltwise/test_elt_binary.py -v
```

## Notes

- All new block variants are marked as "WORK IN PROGRESS - Use with caution"
- Block sizes are compile-time template parameters for optimization
- DEST capacity is enforced with `static_assert(Ht * Wt <= 16)`
- Block variants conform to Compute API Contract (*_block pattern)
- Result stays in DEST for SFPU fusion or further operations
- Companion `pack_*_block` functions available for packing results to L1
