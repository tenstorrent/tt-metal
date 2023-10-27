/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef TT_ETH_API_H
#define TT_ETH_API_H

#include <stdint.h>

#include "eth_ss_regs.h"

#define ETH_WRITE_REG(addr, val) ((*((volatile uint32_t *)((addr)))) = (val))
#define ETH_READ_REG(addr) (*((volatile uint32_t *)((addr))))

void eth_txq_reg_write(uint32_t qnum, uint32_t offset, uint32_t val);

uint32_t eth_txq_reg_read(uint32_t qnum, uint32_t offset);

void eth_send_packet(uint32_t q_num, uint32_t src_word_addr, uint32_t dest_word_addr, uint32_t num_words);

void eth_write_remote_reg(uint32_t q_num, uint32_t reg_addr, uint32_t val);

#endif
