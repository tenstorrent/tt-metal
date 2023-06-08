#pragma once
#include "third_party/taskflow/taskflow/taskflow.hpp"
#include <thread>

namespace tt {
    namespace tt_metal{
        static const size_t EXECUTOR_NTHREADS = std::getenv("TT_METAL_THREADCOUNT") ? std::stoi( std::getenv("TT_METAL_THREADCOUNT") ) : std::thread::hardware_concurrency();

        using Executor = tf::Executor;
        using ExecTask = tf::Task;
        static Executor& GetExecutor() {
            static Executor exec(EXECUTOR_NTHREADS);
            return exec;
        }
    }
}
