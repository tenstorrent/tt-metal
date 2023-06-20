#pragma once

#include <libs/tensor/tensor.hpp>
#include "tt_dnn/op_library/auto_format.hpp"
#include "tt_dnn/op_library/operation.hpp"

namespace tt::tt_metal {

namespace operation_cache {

namespace detail {

struct OperationCache {
    operation::ProgramWithCallbacks& get_or_create(
        const operation::Operation& op,
        const std::vector<std::reference_wrapper<const Tensor>> &input_tensors,
        std::vector<Tensor> &output_tensors,
        Device* device
    ) {
        auto program_hash = op.compute_program_hash(input_tensors);
        if (this->cache_.count(program_hash) == 1) {
            tt::log_info(tt::LogOp, "Operation Cache: HIT - Getting program from the cache \"{}\"", program_hash);
            auto& program = this->cache_.at(program_hash);
            return program;
        } else {
            tt::log_info(tt::LogOp, "Operation Cache: MISS - Compiling new program \"{}\"", program_hash);
            this->cache_[program_hash] = op.create_program(input_tensors, output_tensors);
            auto& program = this->cache_[program_hash].program;
            tt_metal::CompileProgram(device, program);
            return this->cache_[program_hash];
        }
    }

    void enable() {
        this->is_enabled_ = true;
    }

    void disable() {
        this->is_enabled_ = false;
    }

    bool is_enabled() const {
        return this->is_enabled_;
    }

    void clear() {
        this->cache_.clear();
    }

    std::size_t num_cached_programs() const {
        return this->cache_.size();
    }

    private:
        bool is_enabled_ = false;
        std::unordered_map<operation::Hash, operation::ProgramWithCallbacks> cache_{};
};

inline OperationCache OPERATION_CACHE{};

}

template<typename ... Args>
static operation::ProgramWithCallbacks& get_or_create(Args&& ... args) {
    return detail::OPERATION_CACHE.get_or_create(std::forward<Args>(args)...);
}

static bool is_enabled() {
    return detail::OPERATION_CACHE.is_enabled();
}

static void enable() {
    tt::log_info(tt::LogOp, "Operation Cache: enabled.");
    detail::OPERATION_CACHE.enable();
}

static void disable_and_clear() {
    tt::log_info(tt::LogOp, "Operation Cache: disabled and cleared.");
    detail::OPERATION_CACHE.disable();
    detail::OPERATION_CACHE.clear();
}

static std::size_t num_cached_programs() {
    return detail::OPERATION_CACHE.num_cached_programs();
}

}

}
