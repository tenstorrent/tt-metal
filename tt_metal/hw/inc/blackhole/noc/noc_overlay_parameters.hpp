// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// AUTO_GENERATED! DO NOT MODIFY!                                                                                              //
//                                                                                                                             //
// Please run                                                                                                                  //
//                                                                                                                             //
// (echo '<% type=:svh_header %>' && cat noc_overlay_parameters.erb) | erb -T - > ../rtl/overlay/tt_noc_overlay_params.svh   //
// (echo '<% type=:c_header %>' && cat noc_overlay_parameters.erb) | erb -T - > noc_overlay_parameters.h                     //
// (echo '<% type=:cpp_header %>' && cat noc_overlay_parameters.erb) | erb -T - > noc_overlay_parameters.hpp                 //
// (echo '<% type=:rb_header %>' && cat noc_overlay_parameters.erb) | erb -T - > noc_overlay_parameters.rb                   //
// Open noc_overlay_parameters.hpp and move static class varaible definitions to noc_overlay_parameters.cpp                    //
// overriding existing ones.                                                                                                   //
//                                                                                                                             //
// to regenerate                                                                                                               //
//                                                                                                                             //
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>

#ifndef NOC_OVERLAY_PARAMETERS_BASIC_H
#define NOC_OVERLAY_PARAMETERS_BASIC_H

#define NOC_NUM_STREAMS 64
#define ETH_NOC_NUM_STREAMS 32

#define NUM_MCAST_STREAM_ID_START 0
#define NUM_MCAST_STREAM_ID_END   3
#define NUM_RECEIVER_ENDPOINT_STREAM_ID_START 4
#define NUM_RECEIVER_ENDPOINT_STREAM_ID_END   5
#define NUM_REMOTE_RECEIVER_STREAM_ID_START 0
#define NUM_REMOTE_RECEIVER_STREAM_ID_END 63
#define RECEIVER_ENDPOINT_STREAM_MSG_GROUP_SIZE 4
#define RECEIVER_ENDPOINT_STREAM_MSG_INFO_FIFO_GROUPS     4
#define NON_RECEIVER_ENDPOINT_STREAM_MSG_INFO_FIFO_GROUPS 2
#define DEST_READY_COMMON_CACHE_NUM_ENTRIES 24
#define DEST_READY_MCAST_CACHE_NUM_ENTRIES 8

#define NOC_OVERLAY_START_ADDR     0xFFB40000
#define NOC_STREAM_REG_SPACE_SIZE  0x1000

#define STREAM_REG_ADDR(stream_id, reg_id) ((NOC_OVERLAY_START_ADDR) + (((uint32_t)(stream_id))*(NOC_STREAM_REG_SPACE_SIZE)) + (((uint32_t)(reg_id)) << 2))

#define NUM_NOCS                   2
#define NOC0_REGS_START_ADDR       0xFFB20000
#define NOC1_REGS_START_ADDR       0xFFB30000

#define NCRISC_STREAM_RANGE_1_START 0
#define NCRISC_STREAM_RANGE_1_END   3
#define NCRISC_STREAM_RANGE_2_START 8
#define NCRISC_STREAM_RANGE_2_END   11
#define NCRISC_PIC_CONFIG_PHASE_DEFAULT           0

#ifdef TB_NOC

extern "C" {
#include "noc.h"
#include "noc_api_dpi.h"
}

#else

#define NOC_STREAM_WRITE_REG(stream_id, reg_id, val)  ((*((volatile uint32_t*)(STREAM_REG_ADDR(stream_id, reg_id)))) = (val))
#define NOC_STREAM_READ_REG(stream_id, reg_id)        (*((volatile uint32_t*)(STREAM_REG_ADDR(stream_id, reg_id))))

#define NOC_STREAM_WRITE_REG_FIELD(stream_id, reg_id, field, val) (NOC_STREAM_WRITE_REG(stream_id, reg_id, ((NOC_STREAM_READ_REG(stream_id, reg_id) & ~((1 << field##_WIDTH) - 1)) | ((val & ((1 << field##_WIDTH) - 1)) << field))))
#define NOC_STREAM_READ_REG_FIELD(stream_id, reg_id, field)       ((NOC_STREAM_READ_REG(stream_id, reg_id) >> field) & ((1 << field##_WIDTH) - 1))

#define NOC_WRITE_REG(addr, val) ((*((volatile uint32_t*)(addr)))) = (val)
#define NOC_READ_REG(addr)       (*((volatile uint32_t*)(addr)))

#endif


#define NOC_ID_WIDTH     6
#define STREAM_ID_WIDTH  6

#define DEST_CNT_WIDTH   6
#define NOC_NUM_WIDTH     1

#define STREAM_REG_INDEX_WIDTH 9
#define STREAM_REG_CFG_DATA_WIDTH 24

#define MEM_WORD_WIDTH 16
#define MEM_WORD_ADDR_WIDTH 17

#define MEM_WORD_BIT_OFFSET_WIDTH 7

#define MSG_INFO_BUF_SIZE_WORDS 256
#define MSG_INFO_BUF_SIZE_BITS  8
#define MSG_INFO_BUF_SIZE_POW_BITS 3
#define MSG_INFO_BUF_SIZE_WORDS_WIDTH (MSG_INFO_BUF_SIZE_BITS + 1)

#define GLOBAL_OFFSET_TABLE_SIZE 8
#define GLOBAL_OFFSET_TABLE_SIZE_WIDTH 3

#endif

namespace Noc {

typedef struct OverlayField_ {
    std::string name;
    std::uint32_t offset;
    std::uint32_t width;
    std::string description;
} OverlayField;

typedef struct OverlayReg_ {
    std::string name;
    std::uint32_t index;
    std::unordered_map<std::string, std::uint32_t> fields_by_name;
    std::unordered_map<std::uint32_t, std::uint32_t> fields_by_offset;
    std::vector<OverlayField> fields;
    std::string description;
} OverlayReg;

// OverLayParams
class OLP {
    private:
        static const std::unordered_map<std::string, std::uint32_t> registers_by_name;
        static const std::unordered_map<std::uint32_t, std::uint32_t> registers_by_index;
        static const std::vector<OverlayReg> registers;
        static const std::unordered_map<std::string, std::uint32_t> fields_by_name;
        static const std::vector<OverlayField> fields;

    private:
        // Disallow creating an instance of this object
        OLP() {}

    public:
        static bool HasReg(std::string label)
        {
            return registers_by_name.count(label) >= 1;
        }

        // There might be multiple registers with the same index
        // If so a register you didnt intend to access might be accessed.
        // Use accessor based on label if possible
        static bool HasReg(std::uint32_t index)
        {
            return registers_by_index.count(index) >= 1;
        }

        static const std::vector<OverlayReg>& GetAllRegs()
        {
            return registers;
        }

        // There might be multiple registers with the same index
        // If so a register you didnt intend to access might be accessed.
        // Use accessor based on label if possible
        static std::string RegName(std::uint32_t index)
        {
            if (HasReg(index))
                return registers[registers_by_index.at(index)].name;
            else
                throw std::runtime_error("Non-existant overlay register index: " + std::to_string(index));
        }

        static std::uint32_t RegIdx(std::string label)
        {
            if (HasReg(label))
                return registers[registers_by_name.at(label)].index;
            else
                throw std::runtime_error("Non-existant overlay register: " + std::string(label));
        }

        static std::string RegInfo(std::string label)
        {
            if (HasReg(label))
                return registers[registers_by_name.at(label)].description;
            else
                throw std::runtime_error("Non-existant overlay register: " + std::string(label));
        }

        ////////////////////////////////////

        static bool HasFld(std::string label)
        {
            return fields_by_name.count(label) >= 1;
        }

        static const std::vector<OverlayField>& GetAllFlds()
        {
            return fields;
        }

        static std::uint32_t FldOff(std::string label)
        {
            if (HasFld(label))
                return fields[fields_by_name.at(label)].offset;
            else
                throw std::runtime_error("Non-existant overlay field: " + std::string(label));
        }

        static std::uint32_t FldW(std::string label)
        {
            if (HasFld(label))
                return fields[fields_by_name.at(label)].width;
            else
                throw std::runtime_error("Non-existant overlay field: " + std::string(label));
        }

        static std::string FldInfo(std::string label)
        {
            if (HasFld(label))
                return fields[fields_by_name.at(label)].description;
            else
                throw std::runtime_error("Non-existant overlay field: " + std::string(label));
        }

        ////////////////////////////////////

        static bool HasFld(std::string reg_label, std::string field_label)
        {
            return HasReg(reg_label) &&
                   (registers[registers_by_name.at(reg_label)].fields_by_name.count(field_label) >= 1);
        }

        // There might be multiple registers(fields) with the same index(offset)
        // If so a register(field) you didnt intend to access might be accessed.
        // Use accessor based on label if possible
        static bool HasFld(std::uint32_t reg_index, std::uint32_t field_offset)
        {
            return HasReg(reg_index) &&
                   (registers[registers_by_index.at(reg_index)].fields_by_offset.count(field_offset) >= 1);
        }

        static const std::vector<OverlayField>& GetAllFlds(std::string reg_label)
        {
            if (HasReg(reg_label)) {
                return registers[registers_by_name.at(reg_label)].fields;
            } else {
                throw std::runtime_error("Non-existant overlay register: " + std::string(reg_label));
            }
        }

        // There might be multiple registers(fields) with the same index(offset)
        // If so a register(field) you didnt intend to access might be accessed.
        // Use accessor based on label if possible
        static const std::vector<OverlayField>& GetAllFlds(std::uint32_t reg_index)
        {
            if (HasReg(reg_index)) {
                return registers[registers_by_index.at(reg_index)].fields;
            } else {
                throw std::runtime_error("Non-existant overlay register index: " + std::to_string(reg_index));
            }
        }

        // There might be multiple registers(fields) with the same index(offset)
        // If so a register(field) you didnt intend to access might be accessed.
        // Use accessor based on label if possible
        static std::string FldName(std::uint32_t reg_index, std::uint32_t field_offset)
        {
            if (HasFld(reg_index, field_offset)) {
                auto field_tmp = registers[registers_by_index.at(reg_index)].fields;
                auto index_field_temp = registers[registers_by_index.at(reg_index)].fields_by_offset.at(field_offset);
                return field_tmp[index_field_temp].name;
            } else {
                throw std::runtime_error("Non-existant overlay register field (index, offset): " + std::to_string(reg_index) + ", " + std::to_string(field_offset));
            }
        }

        static std::uint32_t FldOff(std::string reg_label, std::string field_label)
        {
            if (HasFld(reg_label, field_label)) {
                auto field_tmp = registers[registers_by_name.at(reg_label)].fields;
                auto index_field_temp = registers[registers_by_name.at(reg_label)].fields_by_name.at(field_label);
                return field_tmp[index_field_temp].offset;
            } else {
                throw std::runtime_error("Non-existant overlay register field: " + std::string(reg_label) + ", " + std::string(field_label));
            }
        }

        static std::uint32_t FldW(std::string reg_label, std::string field_label)
        {
            if (HasFld(reg_label, field_label)) {
                auto field_tmp = registers[registers_by_name.at(reg_label)].fields;
                auto index_field_temp = registers[registers_by_name.at(reg_label)].fields_by_name.at(field_label);
                return field_tmp[index_field_temp].width;
            } else {
                throw std::runtime_error("Non-existant overlay register field: " + std::string(reg_label) + ", " + std::string(field_label));
            }
        }

        // There might be multiple registers(fields) with the same index(offset)
        // If so a register(field) you didnt intend to access might be accessed.
        // Use accessor based on label if possible
        static std::uint32_t FldW(std::uint32_t reg_index, std::uint32_t field_offset)
        {
            if (HasFld(reg_index, field_offset)) {
                auto field_tmp = registers[registers_by_index.at(reg_index)].fields;
                auto index_field_temp = registers[registers_by_index.at(reg_index)].fields_by_offset.at(field_offset);
                return field_tmp[index_field_temp].width;
            } else {
                throw std::runtime_error("Non-existant overlay register field (index, offset): " + std::to_string(reg_index) + ", " + std::to_string(field_offset));
            }
        }

        static std::string FldInfo(std::string reg_label, std::string field_label)
        {
            if (HasFld(reg_label, field_label)) {
                auto field_tmp = registers[registers_by_name.at(reg_label)].fields;
                auto index_field_temp = registers[registers_by_name.at(reg_label)].fields_by_name.at(field_label);
                return field_tmp[index_field_temp].description;
            } else {
                throw std::runtime_error("Non-existant overlay register field: " + std::string(reg_label) + ", " + std::string(field_label));
            }
        }

};

}
