EnqueueWriteBuffer
==================

.. doxygenfunction:: tt::tt_metal::v0::EnqueueWriteBuffer(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer> > buffer, std::vector<uint32_t>& src, bool blocking, tt::stl::Span<const uint32_t> sub_device_ids)
.. doxygenfunction:: tt::tt_metal::v0::EnqueueWriteBuffer(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer> > buffer, HostDataType src, bool blocking, tt::stl::Span<const uint32_t> sub_device_ids)
