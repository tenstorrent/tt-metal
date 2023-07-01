CreateCircularBuffer
=====================

When multiple buffer indices use the same address space, API that accepts a set of buffer indices must be used to ensure that the same address space is not requested from the allocator.

.. doxygenfunction:: CreateCircularBuffer(Program &program, uint32_t buffer_index, const CoreCoord &core, uint32_t num_tiles, uint32_t size_in_bytes, DataFormat data_format, std::optional<uint32_t> = std::nullopt)

.. doxygenfunction:: CreateCircularBuffers(Program &program, uint32_t buffer_index, const CoreRange &core_range, uint32_t num_tiles, uint32_t size_in_bytes, DataFormat data_format, std::optional<uint32_t> = std::nullopt)

.. doxygenfunction:: CreateCircularBuffers(Program &program, uint32_t buffer_index, const CoreRangeSet &core_range_set, uint32_t num_tiles, uint32_t size_in_bytes, DataFormat data_format, std::optional<uint32_t> = std::nullopt)

.. doxygenfunction:: CreateCircularBuffers(Program &program, const std::set<uint32_t> &buffer_indices, const CoreRangeSet &core_range_set, uint32_t num_tiles, uint32_t size_in_bytes, DataFormat data_format, std::optional<uint32_t> = std::nullopt)
