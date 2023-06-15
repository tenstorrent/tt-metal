#pragma once

#include <libs/tensor/tensor.hpp>
#include "tt_dnn/op_library/auto_format.hpp"
#include "tt_dnn/op_library/operation.hpp"

namespace tt::tt_metal {

namespace program_cache {

namespace detail {

struct ProgramCache {
    Program& get_or_create(
        const operation::Operation& op,
        const std::vector<std::reference_wrapper<const Tensor>> &input_tensors,
        std::vector<Tensor> &output_tensors,
        Device* device
    ) {
        auto program_hash = op.compute_program_hash(input_tensors);
        if (this->hash_to_program_.count(program_hash) == 1) {
            tt::log_info(tt::LogOp, "Program Cache: HIT - Getting program from the cache ({})", program_hash);
            auto& program = this->hash_to_program_.at(program_hash);
            return program;
        } else {
            tt::log_info(tt::LogOp, "Program Cache: MISS - Compiling new program ({})", program_hash);
            this->hash_to_program_[program_hash] = op.create_program(input_tensors, output_tensors);
            auto& program = this->hash_to_program_[program_hash];
            tt_metal::CompileProgram(device, program);
            return program;
        }
    }

    void enable() {
        this->is_enabled_ = true;
    }

    bool is_enabled() {
        return this->is_enabled_;
    }

    std::size_t num_cached_programs() {
        return this->hash_to_program_.size();
    }

    private:
        bool is_enabled_ = false;
        std::unordered_map<ProgramHash, Program> hash_to_program_{};
};

inline ProgramCache PROGRAM_CACHE{};

}

template<typename ... Args>
static Program& get_or_create(Args&& ... args) {
    return detail::PROGRAM_CACHE.get_or_create(std::forward<Args>(args)...);
}

static bool is_enabled() {
    return detail::PROGRAM_CACHE.is_enabled();
}

static void enable() {
    tt::log_info(tt::LogOp, "Program Cache: enabled.");
    detail::PROGRAM_CACHE.enable();
}

static std::size_t num_cached_programs() {
    return detail::PROGRAM_CACHE.num_cached_programs();
}

}

}
