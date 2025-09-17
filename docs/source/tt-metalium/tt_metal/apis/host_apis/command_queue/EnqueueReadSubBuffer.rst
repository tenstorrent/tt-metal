EnqueueReadSubBuffer
====================

.. doxygenfunction:: tt::tt_metal::EnqueueReadSubBuffer(CommandQueue& cq, const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer> > &buffer, void* dst, const BufferRegion& region, bool blocking)
.. doxygenfunction:: tt::tt_metal::EnqueueReadSubBuffer(CommandQueue& cq, Buffer& buffer, std::vector<DType>& dst, const BufferRegion& region, bool blocking)
