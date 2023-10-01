/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace tt::tt_metal::detail
{

    struct compile_state_condition_t {
        std::mutex sem_mutex;
        std::condition_variable condition;
        bool done = false; // Initialized as locked.

        void release() {
            {
                unique_lock<mutex> lock(this->sem_mutex);
                this->done = true;
            }
            this->condition.notify_all();
        }

        void acquire() {
            unique_lock<mutex> lock(this->sem_mutex);
            while (not done)
                this->condition.wait(lock);
        }
    };

    struct HashLookup {
    static HashLookup& inst() {
        static HashLookup inst_;
        return inst_;
    }

    bool exists(size_t khash) {
        unique_lock<mutex> lock(mutex_);
        return hashes_.find(khash) != hashes_.end();
    }
    bool add(size_t khash) {
        unique_lock<mutex> lock(mutex_);
        bool ret = false;
        if (hashes_.find(khash) == hashes_.end() ){
            hashes_.emplace(khash, std::make_unique<compile_state_condition_t>());
            ret = true;
        }
        return ret;
    }

    void clear() {
        unique_lock<mutex> lock(mutex_);
        hashes_.clear();
    }

    void mark_compilation_complete(size_t khash) {
        hashes_.at(khash)->release();
    }

    void wait_for_compilation_complete(size_t khash) {
        hashes_.at(khash)->acquire();
    }

    private:
        std::mutex mutex_;
        // Tracks whether a kernel with the given hash has been compiled
        // Existence of a hash in this map indicates that binary does not need to be compiled
        // compile_state_condition_t is used to indicate compilation state and blocks all threads
        //  (via wait_for_compilation_complete) until they are notified that compilation is complete (via mark_compilation_complete)
        std::unordered_map<size_t, std::unique_ptr<compile_state_condition_t>> hashes_;
    };


    /**
     * Clear the current kernel compilation cache.
     *
     * Return value: void
     */
    inline void ClearKernelCache()
    {
        detail::HashLookup::inst().clear();
    }
}
