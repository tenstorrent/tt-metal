| Operation Name | Operation Factory or Python Entry Point |
|----------------|-----------------------------------------|
| EXP | ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp |
| ADD (binary_ng) | ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp |
| ADD (legacy binary sfpu) | ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp |
| WHERE | ttnn/cpp/ttnn/operations/eltwise/ternary/device/ternary_program_factory.cpp |
| TYPECAST (interleaved) | ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_program_factory.cpp |
| TYPECAST (sharded) | ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_sharded_program_factory.cpp |
| DROPOUT | ttnn/cpp/ttnn/operations/experimental/dropout/device/dropout_program_factory.cpp |
| ERFINV | ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp |
| GCD | ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp |
| LOGADDEXP | ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp |
| MAX_POOL_WITH_INDICES | ttnn/cpp/ttnn/operations/pool/generic/device/pool_multi_core_program_factory.cpp |
