EnqueueReadSubBuffer
====================

.. doxygenfunction:: tt::tt_metal::v0::EnqueueReadSubBuffer(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer> > buffer, void* dst, const BufferRegion region, bool blocking, tt::stl::Span<const SubDeviceId> sub_device_ids)
.. doxygenfunction:: tt::tt_metal::v0::EnqueueReadSubBuffer(CommandQueue& cq, Buffer& buffer, std::vector<DType>& dst, const BufferRegion region, bool blocking, tt::stl::Span<const SubDeviceId> sub_device_ids)
