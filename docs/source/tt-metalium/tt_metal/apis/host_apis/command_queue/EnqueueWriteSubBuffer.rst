EnqueueWriteSubBuffer
=====================

.. doxygenfunction:: tt::tt_metal::v0::EnqueueWriteSubBuffer(CommandQueue& cq, const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer> >& buffer, HostDataType src, const BufferRegion& region, bool blocking)
.. doxygenfunction:: tt::tt_metal::v0::EnqueueWriteSubBuffer(CommandQueue& cq, const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer> >& buffer, std::vector<DType>& src, const BufferRegion& region, bool blocking)
