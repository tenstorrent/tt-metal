# Handling Infinity, NaN and denormal numbers in Tensix compute

Doing any math with `NaN` or `+/-Inf` isn't likely to produce good and reliable results so no special treatment for those numbers has been added in TT hardware. However, Tensix FPU and SFPU compute engines do have the ability to detect `NaN`/ `Inf` numbers in FPU and sFPU ops, so the programmer has the ability to see that somewhere in the calculations `NaN` / `Inf` has been found and discard results (or work on fixing them not to appear).

## Representation

Representation of special values in TT hardware is illustrated in the following table:
```
Special Value   |   Format   |    Value
-------------------------------------------
+ NaN           |    FP32    | 0x7FFF_FFFF
- NaN           |    FP32    | 0xFFFF_FFFF
+ Inf           |    FP32    | 0x7F80_0000
- Inf           |    FP32    | 0xFF80_0000
+ NaN           |  BFLOAT16  |   0x7FFF
- NaN           |  BFLOAT16  |   0xFFFF
+ Inf           |  BFLOAT16  |   0x7F80
- Inf           |  BFLOAT16  |   0xFF80
+ NaN           |  FLOAT16   |   0x7FFF
- NaN           |  FLOAT16   |   0xFFFF
+ Inf           |  FLOAT16   |   0x7C00
- Inf           |  FLOAT16   |   0xFC00
denormals       |    all     |     0x0
--------------------------------------------
```

## Detection of special numbers and debugging

TT Hardware is not fully IEEE compliant, however, some operations are following the standard when it comes to special number handling:

1. For FPU ops (add_tiles, mul_tiles or matmul_tiles):
```
       Operation      |   Result
-----------------------------------
       ±Inf × ±Inf​    |    ±Inf​
    ±finite × ±Inf​    |    ±Inf​
        Inf + Inf​     |    +Inf​
       -Inf – Inf​     |    –Inf​
```
2. For SFPU ops:
```
       Operation      |   Result
-----------------------------------
       ±Inf × ±Inf​    |    ±Inf​
    ±finite × ±Inf​    |    ±Inf​
        Inf + Inf​     |    +Inf​
       -Inf – Inf​     |    –Inf​
          0 × Inf​     |     NaN
       +Inf - Inf​     |     NaN
```

The operations not listed in the tables above will treat `NaN`/`Inf` numbers just as any other ordinary number, therefore the result may or may not be a special number designated by the standard. However, Tensix compute hardware can detect if any of the result value have ever become one of the special numbers, thus providing the information to the programmer that one of those values has been "seen".
`NaN` and `+/-Inf` numbers are detected at the output of FPU and SFPU operations by setting status flags. The flags are ORed between all FPU or all SFPU lanes and it is not possible to determine from which FPU/SFPU lane the number came from. The following flags are supported:
 - status_bit[6] - Any FPU lane exponent underflow (denorm number detected)
 - status_bit[5] - Any FPU lane infinity/overflow detected
 - status_bit[4] - Any FPU lane int32 saturation detected (when doing int32 accumulation)
 - status_bit[3] - Any SFPU lane detected NaN
 - status_bit[2] - Any SFPU lane detected infinity
 - status_bit[1] - Any SFPU lane detected denorm
 - status_bit[0] - Any SFPU lane detected overflow (through int32 addition)

Each Tensix core has a register that programmer can read at any point in the kernel to check if any of the previous operations caused a special number to be seen. The status flags are sticky and will stay detected until cleared explicitly by the programmer. The register is accessible over NOC, from host/other Tensix core, or through risc cores in the same Tensix core:
```
#define RISCV_DEBUG_REG_FPU_STICKY_BITS         (0xffb1200 | 0x0B4)
```

Compute API funcitons are provided to read and clear flags:
```
  TODO: Add API functions
```

Given that detection of `NaN`/`Inf` is only done at the output of FPU/SFPU operations, there are some cases of special numbers which can be undetected:
1. Special numbers appeared as input operands
2. Special numbers appeared as intermediate results inside FPU operations that don't propage to output

Debugging undetected cases can be hard and we plan to address that in future architectures (depending on area/perf cost).

Some ideas for Wormhole:
- to debug 1. -
   - all inputs to the device should be filtered from the host not to have any Inf/NaNs
   - Given that FPU/SFPU hardware detects if NaN/Inf numbers are detected at output and device inputs are filtered, detection at input isn't really necessary given that any special numbers would've already been caught at output of previous ops
- to debug 2. -
   - To the best of our knowledge, activations and weights are numbers with normal distribution (mean = 0.0, stddev = 1.0)
   - Numbers with really big magnitude are unlikely to be correct, so operations with `Inf/NaN` and `finite` that don't propagate special numbers properly will, sooner or later, likely yeild numbers with extremely big magnitude
   - Cases like `Inf/NaN` + `normal magnitude finite` would produce output numbers would have really big magnitude and significantly stand out as incorrect. The debug tools allow us to DPRINT results after every tile is calculated so it should be possible to detect/spot erroneous numbers even though they are undetected.
   - Cases where `Inf/NaN` + `really big magnitude finite` could be entirely cancelled and fall into the "valid" range, but it shouldn't be possible to get `really big magnitude finite` number and still being correct. Debug tools could be used to detect those `really big magnitude finite` numbers and detect errors
