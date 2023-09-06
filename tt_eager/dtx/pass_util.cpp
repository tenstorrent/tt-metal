// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dtx.hpp"
#include "util.hpp"
#include "util_vector_of_ints.hpp"

// Copies all attributes from producer group to the consumer group
// Today, the list of attributes is very short :) but it will get bigger as we add functionality.
// Ex)
//     1. WH device, buffer address, core ID, etc...
void inherit_group_attributes_from_producer(TensorPairGroup * producer, TensorPairGroup * consumer) {
    consumer->core = {};
    consumer->core.push_back(producer->core[1]);
    consumer->core.push_back(producer->core[0]);
}
