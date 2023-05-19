#pragma once

#include <stdlib.h>
#include <iostream>

struct OpEnvConfig {
    static void update_num_cores(uint32_t* num_cores) {
        const char *ttnc = getenv("TT_FORCE_NUMCORES");
        if ( ttnc != nullptr) {
            int new_val = std::stoi(ttnc);
            if (new_val != 0)
                *num_cores = new_val;
            std::cout << "=== num_cores after update from env[\"TT_FORCE_NUMCORES\"]=" << *num_cores << std::endl;
        }
    }

    static void update_profile(bool* profile) {
        const char *ttprof = getenv("TT_PROFILE");
        if (ttprof != nullptr) {
            *profile = true;
            std::cout << "=== profile flag after update from ENV[\"TT_PROFILE\"]=" << *profile << std::endl;
        }
    }

    static void update_block_size(uint32_t* block_size) {
        if (getenv("TT_BLOCK_SIZE") != nullptr) {
            *block_size = std::stoi( getenv("TT_BLOCK_SIZE") );
            std::cout << "=== block_size after update from ENV[\"TT_BLOCK_SIZE\"]=" << *block_size << std::endl;
        }
    }
};
