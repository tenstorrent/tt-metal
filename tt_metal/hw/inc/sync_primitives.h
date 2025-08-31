#pragma once

#include <cstdint>
#include <bitset>

#include "core_config.h"
#include "dataflow_api_addrgen.h"
#include "debug/assert.h"
#include "debug/sanitize_noc.h"
#include "debug/waypoint.h"
#include "noc_parameters.h"
#include "tools/profiler/noc_event_profiler.hpp"

namespace experimental {

class SemaphoreId {
    explicit constexpr SemaphoreId(uint8_t id) : value(id) {}
    
    template<uint8_t N>
    static constexpr SemaphoreId from_constant() {
        static_assert(N < NUM_SEMAPHORES, "Semaphore ID exceeds maximum");
        return SemaphoreId(N);
    }
        
    constexpr bool operator==(const SemaphoreId& other) const { return value == other.value; }
    constexpr bool operator!=(const SemaphoreId& other) const { return value != other.value; }
    constexpr bool operator<(const SemaphoreId& other) const { return value < other.value; }
    
    // Invalid ID constant
    static constexpr SemaphoreId invalid() { return SemaphoreId(UINT8_MAX); }
    constexpr bool is_valid() const { return value_ != UINT8_MAX; }

    uint8_t value = UINT8_MAX;
};

template <ProgrammableCoreType type = ProgrammableCoreType::TENSIX>
class Semaphore {
public:
    explicit Semaphore(SemaphoreId id) : id_(id) {
        ASSERT(id.value() < NUM_SEMAPHORES);
        ASSERT(!allocated_ids_mask_.test(id.value()));
        allocated_ids_mask_.set(id.value());
        addr_ = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            (uint32_t)sem_l1_base[static_cast<int>(type)] + id.value * L1_ALIGNMENT);
    }

    ~Semaphore() {
        if (id_.is_valid()) {
            allocated_ids_mask_.reset(id_.value);
        }
    }

    Semaphore(const Semaphore&) = delete;
    Semaphore& operator=(const Semaphore&) = delete;

    Semaphore(Semaphore&& other) noexcept : id_(other.id_) {
        other.id_ = SemaphoreId::invalid();
    }

    Semaphore& operator=(Semaphore&& other) noexcept {
        if (this != &other) {
            if (id_.is_valid()) {
                allocated_ids_mask_.reset(id_.value);
            }
            
            id_ = other.id_;
            other.id_ = SemaphoreId::invalid();
        }
        return *this;
    }

    FORCE_INLINE SemaphoreId id() const { return id_; }

    // Local operations
    FORCE_INLINE void wait(uint32_t val) const {
        RECORD_NOC_EVENT(NocEventType::SEMAPHORE_WAIT);

        WAYPOINT("NSW");
        do {
            invalidate_l1_cache();
        } while ((*addr_) != val);
        WAYPOINT("NSD");
    }

    FORCE_INLINE void wait_min(uint32_t val) const {
        RECORD_NOC_EVENT(NocEventType::SEMAPHORE_WAIT);

        WAYPOINT("NSMW");
        do {
            invalidate_l1_cache();
        } while ((*addr_) < val);
        WAYPOINT("NSMD");
    }

    FORCE_INLINE void set(uint32_t val) {
        RECORD_NOC_EVENT(NocEventType::SEMAPHORE_SET);
        (*addr_) = val;
    }

private:
    SemaphoreId id_;
    volatile uint32_t tt_l1_ptr* addr_;
    
    static inline std::bitset<NUM_SEMAPHORES> allocated_ids_mask_;
};

// Remote semaphore operations - operates on semaphores on other cores
template <ProgrammableCoreType type = ProgrammableCoreType::TENSIX>
class RemoteSemaphore {
public:
    explicit RemoteSemaphore(SemaphoreId target_semaphore_id) : target_semaphore_id_(target_semaphore_id) {
        ASSERT(target_semaphore_id.value < NUM_SEMAPHORES);
        target_addr_ = reinterpret_cast<uint32_t>(
            sem_l1_base[static_cast<int>(type)] + target_semaphore_id.value * L1_ALIGNMENT);
    }

    template <bool posted = false>
    FORCE_INLINE void inc(uint8_t dst_x, uint8_t dst_y, uint32_t incr, uint8_t noc_id = noc_index, uint8_t vc = NOC_UNICAST_WRITE_VC) const {
        uint64_t dst_noc_addr = get_noc_addr(dst_x, dst_y, (uint32_t)target_addr_);

        WAYPOINT("NSIW");
        DEBUG_SANITIZE_NOC_ADDR(noc_id, dst_noc_addr, 4);
        DEBUG_INSERT_DELAY(TransactionAtomic);
        noc_fast_atomic_increment<noc_mode>(
            noc_id,
            write_at_cmd_buf,
            dst_noc_addr,
            vc,
            incr,
            31 /*wrap*/,
            false /*linked*/,
            posted /*posted*/,
            MEM_NOC_ATOMIC_RET_VAL_ADDR);
        WAYPOINT("NSID");
    }

    template <ProgrammableCoreType local_type>
    FORCE_INLINE void set(const Semaphore<local_type> &local_sem, uint8_t dst_x, uint8_t dst_y, uint8_t noc_id = noc_index) const {
        WAYPOINT("NSSW");
        uint32_t local_addr = (uint32_t)sem_l1_base[static_cast<int>(local_type)] + local_sem.id().value * L1_ALIGNMENT;
        uint64_t dst_noc_addr = get_noc_addr(dst_x, dst_y, (uint32_t)target_addr_);
        DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc, dst_noc_addr, local_addr, 4);
        ncrisc_noc_fast_write_any_len<noc_mode>(
            noc_id,
            write_reg_cmd_buf,
            local_addr,
            dst_noc_addr,
            4,
            NOC_UNICAST_WRITE_VC,
            false,
            false,
            1,
            true);
        WAYPOINT("NSSD");
    }

    // Alternative: raw address variant -- users need to lock and unlock around this
    FORCE_INLINE void set(uint32_t local_addr, uint8_t dst_x, uint8_t dst_y, uint8_t noc_id = noc_index) const {
        WAYPOINT("NSSW");
        uint64_t dst_noc_addr = get_noc_addr(dst_x, dst_y, (uint32_t)target_addr_);
        DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc, dst_noc_addr, local_addr, 4);
        ncrisc_noc_fast_write_any_len<noc_mode>(
            noc_id,
            write_reg_cmd_buf,
            local_addr,
            dst_noc_addr,
            4,
            NOC_UNICAST_WRITE_VC,
            false,
            false,
            1,
            true);
        WAYPOINT("NSSD");
    }

    template <ProgrammableCoreType local_type>
    FORCE_INLINE void set_multicast(const Semaphore<local_type> &local_sem, uint8_t dst_start_x, uint8_t dst_start_y, uint8_t dst_end_x, uint8_t dst_end_y, uint32_t num_dests, bool linked = false,uint8_t noc_id = noc_index) const {
        WAYPOINT("NSNW");
        uint32_t local_addr = (uint32_t)sem_l1_base[static_cast<int>(local_type)] + local_sem.id().value * L1_ALIGNMENT;
        uint64_t dst_noc_addr_multicast = get_noc_multicast_addr(dst_start_x, dst_start_y, dst_end_x, dst_end_y, (uint32_t)target_addr_, noc_id);
        DEBUG_SANITIZE_NOC_MULTI_WRITE_TRANSACTION(noc_id, dst_noc_addr_multicast, local_addr, 4);
        // TODO should all mcast that aren't loopback src assert my_x/my_y are not part of target grid
        ncrisc_noc_fast_write_any_len<noc_mode>(
            noc_id,
            write_reg_cmd_buf,
            local_addr,
            dst_noc_addr_multicast,
            4 /*size in bytes*/,
            NOC_MULTICAST_WRITE_VC,
            true,
            linked,
            num_dests,
            true /* multicast_path_reserve */);
        WAYPOINT("NSND");
    }

    FORCE_INLINE void set_multicast(uint32_t local_addr, uint8_t dst_start_x, uint8_t dst_start_y, uint8_t dst_end_x, uint8_t dst_end_y, uint32_t num_dests, bool linked = false,uint8_t noc_id = noc_index) const {
        WAYPOINT("NSNW");
        uint64_t dst_noc_addr_multicast = get_noc_multicast_addr(dst_start_x, dst_start_y, dst_end_x, dst_end_y, (uint32_t)target_addr_, noc_id);
        DEBUG_SANITIZE_NOC_MULTI_WRITE_TRANSACTION(noc_id, dst_noc_addr_multicast, local_addr, 4);
        ncrisc_noc_fast_write_any_len<noc_mode>(
            noc_id,
            write_reg_cmd_buf,
            local_addr,
            dst_noc_addr_multicast,
            4 /*size in bytes*/,
            NOC_MULTICAST_WRITE_VC,
            true,
            linked,
            num_dests,
            true /* multicast_path_reserve */);
        WAYPOINT("NSND");
    }

    template <ProgrammableCoreType local_type>
    FORCE_INLINE void set_multicast_loopback_src(const Semaphore<local_type> &local_sem, uint8_t dst_start_x, uint8_t dst_start_y, uint8_t dst_end_x, uint8_t dst_end_y, uint32_t num_dests, bool linked = false,uint8_t noc_id = noc_index) const {
        WAYPOINT("NSNW");
        uint32_t local_addr = (uint32_t)sem_l1_base[static_cast<int>(local_type)] + local_sem.id().value * L1_ALIGNMENT;
        uint64_t dst_noc_addr_multicast = get_noc_multicast_addr(dst_start_x, dst_start_y, dst_end_x, dst_end_y, (uint32_t)target_addr_, noc_id);
        DEBUG_SANITIZE_NOC_MULTI_WRITE_TRANSACTION(noc_id, dst_noc_addr_multicast, local_addr, 4);
        ncrisc_noc_fast_write_any_len_loopback_src<noc_mode>(
            noc_id,
            write_reg_cmd_buf,
            local_addr,
            dst_noc_addr_multicast,
            4 /*size in bytes*/,
            NOC_MULTICAST_WRITE_VC,
            true,
            linked,
            num_dests,
            true /* multicast_path_reserve */);
        WAYPOINT("NSND");
    }

    FORCE_INLINE void set_multicast_loopback_src(uint32_t local_addr, uint8_t dst_start_x, uint8_t dst_start_y, uint8_t dst_end_x, uint8_t dst_end_y, uint32_t num_dests, bool linked = false,uint8_t noc_id = noc_index) const {
        WAYPOINT("NSNW");
        uint64_t dst_noc_addr_multicast = get_noc_multicast_addr(dst_start_x, dst_start_y, dst_end_x, dst_end_y, (uint32_t)target_addr_, noc_id);
        DEBUG_SANITIZE_NOC_MULTI_WRITE_TRANSACTION(noc_id, dst_noc_addr_multicast, local_addr, 4);
        ncrisc_noc_fast_write_any_len_loopback_src<noc_mode>(
            noc_id,
            write_reg_cmd_buf,
            local_addr,
            dst_noc_addr_multicast,
            4 /*size in bytes*/,
            NOC_MULTICAST_WRITE_VC,
            true,
            linked,
            num_dests,
            true /* multicast_path_reserve */);
        WAYPOINT("NSND");
    }

    SemaphoreId id() const { return target_semaphore_id_; }

private:
    SemaphoreId target_semaphore_id_;
    uint32_t target_addr_;
};

}  // namespace experimental