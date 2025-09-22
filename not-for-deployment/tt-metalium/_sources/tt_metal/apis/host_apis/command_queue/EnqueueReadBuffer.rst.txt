EnqueueReadBuffer
==================

.. doxygenfunction:: tt::tt_metal::EnqueueReadBuffer(CommandQueue &cq, Buffer &buffer, std::vector<DType> &dst, bool blocking)
.. doxygenfunction:: tt::tt_metal::EnqueueReadBuffer(CommandQueue& cq, const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer> > &buffer, void * dst, bool blocking)
