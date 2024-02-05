EnqueueProgram
==============

void EnqueueProgram(CommandQueue& cq, std::variant<std::reference_wrapper<Program>, std::shared_ptr<Program> > program, bool blocking, std::optional<std::reference_wrapper<Trace>> trace);
