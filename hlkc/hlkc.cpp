#include "hlkc.hpp"
#include "staticSingleAssignment.h"
#include "hlkc_cache.hpp"
#include "hlkc_util.h"

// #include <functional>
#include <set>
#include <sstream>

//#include "pre.h"  // not used atm
//#include "inlinerSupport.h" // not used atm

using namespace std;
using namespace Rose;
using namespace SageBuilder;
using namespace SageInterface;

namespace fs = std::filesystem;

typedef Rose_STL_Container<SgNode*> SgNodeContainer;
typedef SgNodeContainer::iterator SgNodeContainerIter;

#define SOURCE_POSITION Sg_File_Info::generateDefaultFileInfoForTransformationNode()

void generate_unpack(SgProject *project, string llk_args_file_name, bool perf_dump_en, string device_name, bool fp32_dest_acc_en) {
    auto hlk_ops = get_hlk_matrix_ops(project);

    //assert(hlk_ops.size() > 0);
    HlkMatrixOp* hlk_matrix_op;
    if (hlk_ops.size() > 0) {
        hlk_matrix_op = hlk_ops[0];
    } else {
        hlk_matrix_op = nullptr;
    }

    bool multi_op = hlk_ops.size() > 1;

    for (auto& op : hlk_ops) {
        if (op->get_op_api() == HlkMatrixOp::OpApi::hlk_mm_tile) {
            hlk_matrix_op = op;
            break;
        }
    }

    vector<HlkSfpuOp*>   hlk_sfpu_ops   = get_hlk_sfpu_op(project);

    bool adv_features_en = DeviceEnum::GRAYSKULL != device_enum_from_string(device_name);

    // Specialize all ops
    for (auto& op : hlk_ops) {
        op->specialize(get_int_arg_vals_for_hlk_op(project, op), adv_features_en, fp32_dest_acc_en);
    }
#if 0
    if (multi_op) {
        for (auto& op : hlk_ops) {
            if (op->get_op_api() == HlkMatrixOp::OpApi::hlk_copy_tile_to_dst) {
                op->specialize_unpack("<BroadcastType::NONE, true>");
                break;
            }
        }
    }
#endif

    // insert the generic LLK unpack header
    insert_header(project, "llk_unpack_common.h");
    remove_header(project, "compute_hlk_api.h");
    // insert the custom unpack header for this op
    for (auto& op : hlk_ops) {
        insert_header(project, op->get_unpack_include_header_str());
    }

    vector<string> func_calls_to_remove = {
        "hlk_wait_for_free_tiles",
        "hlk_push_tiles",
        "hlk_acquire_dst",
        "hlk_release_dst",
        "hlk_pack_tile_to_stream",
        "hlk_pack_relu_tile_to_stream",
        "hlk_pack_shifted_init",
        "hlk_pack_shifted_tile_to_stream",
        "hlk_pack_shifted_relu_tile_to_stream",
        "hlk_reconfig_packer_df",
        "hlk_relu_config"};

    for (auto hlk_sfpu_op: hlk_sfpu_ops)  {
        func_calls_to_remove.push_back(hlk_sfpu_op->get_op_str());
        func_calls_to_remove.push_back(hlk_sfpu_op->get_op_str() + "_init");
    }

    if (not multi_op and hlk_ops.size() > 0) {
        func_calls_to_remove.push_back(hlk_matrix_op->get_op_str() + "_init");
        func_calls_to_remove.push_back(hlk_matrix_op->get_op_str() + "_init_short");
        func_calls_to_remove.push_back(hlk_matrix_op->get_op_str() + "_init_once");
    }
    remove_call_statements(project, func_calls_to_remove);

    replace_call_statements(project, {"hlk_wait_tiles",  "hlk_pop_tiles", "hlk_get_tile"       , "hlk_release_tile"       , "hlk_debug_dump",        "hlk_reconfig_unpacker_df",        "hlk_flush_tiles", "hlk_get_next_op_info"},
                                     {"llk_wait_tiles",  "llk_pop_tiles", "llk_unpack_get_tile", "llk_unpack_release_tile", "llk_unpack_debug_dump", "llk_unpack_reconfig_data_format", "llk_clear_tiles", "llk_get_next_op_info"});

    remove_vars_of_type(project, {"struct ::hlk_pack_shifted_params_t", "struct ::hlk_pack_shifted_state_t"});

    vector<string> hlk_init_list;
    vector<string> llk_init_list;
    vector<SgNode*> hlk_api_func_list;
    vector<int> llk_positions;
    vector<string> hw_func_names;
    vector<SgExpression*> unpack_hw_configure_call_expr_ptr_list;

    patch_main_decl_for_llk(project, "unpack_main");

    if (hlk_ops.size() > 0) {
        insert_outer_loop(project, "hlk_main", perf_dump_en, 0);
    }

    if (std::getenv("HACK_FOR_GRAPH_INTERPRETER") != nullptr) {
        for (auto& op : hlk_ops) {
            replace_hlk_inits_and_insert_hw_configure(project, op, "unpack");
        }
        string hw_func_name = "llk_unpack_AB_hw_configure_disaggregated";
        SgNodeContainer func_call_list = find_function_calls(project);
        SgFunctionDeclaration* main_decl = find_function_declaration(project, "hlk_main");
        SgFunctionDefinition* main_def = main_decl->get_definition();
        SgBasicBlock* body = main_def->get_body();


        SgExprListExp* expr_list = new SgExprListExp(SOURCE_POSITION);
        expr_list->append_expression(
            buildIntVal(0)
        );
        expr_list->append_expression(
            buildIntVal(1)
        );

        SgExprStatement* new_call_statement = build_void_func_call_statement(hw_func_name, expr_list, main_def);
        insertStatementAfter(*(body->get_statements().begin()), new_call_statement);

    } else if (multi_op) {
        for (auto& op : hlk_ops) {
            replace_hlk_inits_and_insert_hw_configure(project, op, "unpack");
        }
    } else if (hlk_ops.size() > 0) {
        hlk_api_func_list = find_function_calls_by_name(project, hlk_ops[0]->get_op_str());
        llk_init_list = {hlk_ops[0]->get_unpack_init_func_str()};
        hw_func_names = {insert_disaggregated_into_hw_configure(hlk_ops[0]->get_unpack_hw_configure_func_str())};
        unpack_hw_configure_call_expr_ptr_list =  get_func_call_sg_expression_arg_list(
            hlk_api_func_list.at(0)->get_parent(),
            hlk_ops[0]->get_unpack_hw_configure_func_hlk_api_expr_positions()
        );
    }

    vector<vector<int>> operand_ids = replace_call_statements_for_unpack_and_get_multiple_op_operand_ids(project, hlk_ops);

    if (not multi_op and hlk_ops.size() > 0) {
        if (operand_ids.at(0).size() == 0) {
            operand_ids = {{0}};
        }
        insert_void_argless_func_call_to_statement_list(
            project, "hlk_main", hlk_ops[0]->get_unpack_init_func_str(), 1
        );

        insert_hw_configure_after_init(
            project,
            "hlk_main",
            hw_func_names,
            llk_init_list,
            operand_ids,
            {unpack_hw_configure_call_expr_ptr_list}
        );
    }

    if (hlk_ops.size() > 0) {
        insert_void_argless_func_call_to_statement_list(project, "hlk_main", "llk_setup_operands", 1);
    }
    insert_namespace(project, "hlk_main", "NAMESPACE");
}

void generate_math(SgProject *project, string llk_args_file_name, bool perf_dump_en,  string device_name, bool fp32_dest_acc_en) {
    vector<HlkOp*> hlk_ops = {};
    auto hlk_matrix_ops = get_hlk_matrix_ops(project);
    for (auto& op : hlk_matrix_ops) {
        hlk_ops.push_back(op);
    }

    HlkMatrixOp* hlk_matrix_op;
    if (hlk_ops.size() > 0) {
        hlk_matrix_op = hlk_matrix_ops[0];
    } else {
        hlk_matrix_op = nullptr;
    }

    vector<HlkSfpuOp*>   hlk_sfpu_ops   = get_hlk_sfpu_op(project);
    HlkSfpuOp* hlk_sfpu_op = hlk_sfpu_ops.empty()  ? NULL : hlk_sfpu_ops.at(0);

    bool multi_op = hlk_matrix_ops.size() > 1;
    bool multi_sfpu_op = hlk_sfpu_ops.size() > 1;

    for (auto& op : hlk_matrix_ops) {
        if (op->get_op_api() == HlkMatrixOp::OpApi::hlk_mm_tile) {
            hlk_matrix_op = op;
            break;
        }
    }

    string dst_mode_str = "";
    if (hlk_ops.size() > 0) {
        dst_mode_str = detect_dst_mode(project, hlk_sfpu_op, hlk_matrix_op);
    }
    bool adv_features_en = DeviceEnum::GRAYSKULL != device_enum_from_string(device_name);
    for (auto& op : hlk_matrix_ops) {
        op->specialize(get_int_arg_vals_for_hlk_op(project, op), adv_features_en, fp32_dest_acc_en);
        op->specialize_math_pack_sync(dst_mode_str, false, adv_features_en, fp32_dest_acc_en);
        op->specialize_math_func_str(dst_mode_str, adv_features_en, fp32_dest_acc_en);
    }

    for (auto hlk_sfpu_op: hlk_sfpu_ops) {
        hlk_ops.push_back(hlk_sfpu_op);
        hlk_sfpu_op->specialize(get_int_arg_vals_for_hlk_op(project, hlk_sfpu_op), adv_features_en, fp32_dest_acc_en);
        hlk_sfpu_op->specialize_math_func_str(dst_mode_str, adv_features_en, fp32_dest_acc_en);
    }

    remove_header(project, "compute_hlk_api.h");
    // insert the generic LLK math header
    insert_header(project, "llk_math_common.h");
    // insert the custom math matrix op header for this op
    for (auto& op : hlk_matrix_ops) {
        insert_header(project, op->get_math_include_header_str());
    }
    // insert the sfpu header is sfpu is used
    for (auto hlk_sfpu_op: hlk_sfpu_ops)  {
        insert_header(project, hlk_sfpu_op->get_math_include_header_str());
    }

    remove_call_statements(
        project,
        {"hlk_wait_tiles",
         "hlk_pop_tiles",
         "hlk_wait_for_free_tiles",
         "hlk_push_tiles",
         "hlk_pack_tile_to_stream",
         "hlk_pack_relu_tile_to_stream",
         "hlk_pack_shifted_init",
         "hlk_pack_shifted_tile_to_stream",
         "hlk_pack_shifted_relu_tile_to_stream"
         "hlk_reconfig_packer_df",
         "hlk_relu_config"});

    remove_vars_of_type(project, {"struct ::hlk_pack_shifted_params_t", "struct ::hlk_pack_shifted_state_t"});

    replace_call_statements_for_math(project, hlk_ops);

    replace_call_statements(project, {"hlk_get_tile"     , "hlk_release_tile"      , "hlk_debug_dump" ,      "hlk_reconfig_unpacker_df",      "hlk_get_next_op_info" },
                                     {"llk_math_get_tile", "llk_math_release_tile" , "llk_math_debug_dump" , "llk_math_reconfig_data_format", "llk_get_next_op_info"});

    patch_main_decl_for_llk(project, "math_main"); // TODO: this probably incomplete (so we keep using "hlk_main" below), I think we need to update the symbol table

    // TODO: These hlk_acquire_dst/hlk_release_dst will likely fail with real multi-op
    if (hlk_matrix_ops.size() > 0) {
        insert_void_argless_func_calls_after_func_call(project, "hlk_main", "hlk_acquire_dst", {hlk_matrix_op->get_math_wait_for_dest_available()});
        insert_void_argless_func_calls_after_func_call(project, "hlk_main", "hlk_release_dst", {hlk_matrix_op->get_math_dest_section_done()});
    }
    remove_call_statements(project, {"hlk_acquire_dst", "hlk_release_dst", "hlk_debug_dump", "hlk_flush_tiles"});

    if (hlk_ops.size() > 0) {
        insert_outer_loop(project, "hlk_main", perf_dump_en, 1); // FIXME: should be using unpack_main, but function "unpack_main" can't be found
    }

    if (multi_op) {
        vector<string> hlk_init_list = {};
        vector<string> llk_init_list = {};
        for (auto& op : hlk_matrix_ops) {
            hlk_init_list.push_back(op->get_op_str() + "_init");
            llk_init_list.push_back(op->get_math_init_func_str());

            hlk_init_list.push_back(op->get_op_str() + "_init_once");
            llk_init_list.push_back(op->get_math_init_func_str());

            hlk_init_list.push_back(op->get_op_str() + "_init_short");
            llk_init_list.push_back(op->get_math_init_func_str());
        }
        replace_call_statements(project, hlk_init_list, llk_init_list);
    } else if (hlk_matrix_ops.size() > 0) {
        remove_call_statements(project, {hlk_matrix_op->get_op_str() + "_init", hlk_matrix_op->get_op_str() + "_init_short", hlk_matrix_op->get_op_str() + "_init_once"});
    }

    if (hlk_matrix_ops.size() > 0) {
        insert_void_argless_func_call_to_statement_list(project, "hlk_main", hlk_matrix_op->get_math_pack_sync_init(), 1);
    }

    if (not multi_op and hlk_matrix_ops.size() > 0) {
        insert_void_argless_func_call_to_statement_list(
            project, "hlk_main", hlk_matrix_op->get_math_init_func_str(), 1);
    }

    for (auto hlk_sfpu_op: hlk_sfpu_ops)  {
        replace_call_statements(project, {hlk_sfpu_op->get_op_str() + "_init"},{hlk_sfpu_op->get_math_init_func_str()});
    }

    if (hlk_sfpu_op != NULL) {
        insert_void_argless_func_call_to_statement_list(project, "hlk_main", hlk_sfpu_op->get_math_init_func_str(), 1);
    }

    // If dropout, we need to do a post-pass to ensure that the dropout probability is converted from float to uint16 format and that
    // scale is fp1_8_7 format
    replace_hlk_dropout_with_llk_int_dropout_and_scale(project);

    insert_namespace(project, "hlk_main", "NAMESPACE");
}

void generate_pack(SgProject *project, string llk_args_file_name, bool perf_dump_en, bool untilize_output, bool pack_microblocks, string device_name, bool fp32_dest_acc_en, uint32_t relu_config, bool pack_l1_acc_en) {
    auto hlk_ops = get_hlk_matrix_ops(project);

    HlkMatrixOp* hlk_matrix_op;
    if (hlk_ops.size() > 0) {
        hlk_matrix_op = hlk_ops[0];
    } else {
        hlk_matrix_op = nullptr;
    }

    bool multi_op = hlk_ops.size() > 1;

    for (auto& op : hlk_ops) {
        if (op->get_op_api() == HlkMatrixOp::OpApi::hlk_mm_tile) {
            hlk_matrix_op = op;
            break;
        }
    }

    vector<HlkSfpuOp*>   hlk_sfpu_ops   = get_hlk_sfpu_op(project);
    HlkSfpuOp* hlk_sfpu_op = hlk_sfpu_ops.empty()  ? NULL : hlk_sfpu_ops.at(0);
    bool adv_features_en = DeviceEnum::GRAYSKULL != device_enum_from_string(device_name);

    string dst_tile_face_layout_str = "DstTileFaceLayout::RowMajor";
    if (hlk_ops.size() > 0) {
        dst_tile_face_layout_str = "DstTileFaceLayout::RowMajor"; //only matmul op outputs tiles with face layout in column major form (0,2,1,3)
    }

    bool const has_pack_shifted_relu = !find_function_calls_by_name(project, "hlk_pack_shifted_relu_tile_to_stream").empty();
    bool const has_pack_shifted = (!find_function_calls_by_name(project, "hlk_pack_shifted_tile_to_stream").empty()) or has_pack_shifted_relu;

    string llk_pack_init_func = has_pack_shifted ? "llk_pack_shifted_init" : "llk_pack_init";
    assert (not (has_pack_shifted and untilize_output) && "Unsupported: llk_pack_shifted_init does not have untilize mode!");
    // TODO: false is the default of second template arg (for llk_pack_init), need a better solution than hard-coding this (jchen)
    //      Currently a bug in ROSE (failure in runAllTests) if only one template is specified, linked to issue #123
    llk_pack_init_func = untilize_output ? llk_pack_init_func + "<true, false, " + dst_tile_face_layout_str + ">" : llk_pack_init_func;

    string dst_mode_str = "";
    if (hlk_ops.size() > 0) {
        dst_mode_str = detect_dst_mode(project, hlk_sfpu_op, hlk_matrix_op);
    }

    for (auto& op : hlk_ops) {
        op->specialize_math_pack_sync(dst_mode_str, untilize_output, adv_features_en, fp32_dest_acc_en);
        op->specialize_pack_func_str(get_int_arg_vals_for_hlk_op(project, op), untilize_output, adv_features_en, fp32_dest_acc_en);
    }

    // insert the generic LLK math header
    insert_header(project, "llk_pack_common.h");
    if (has_pack_shifted) {
       insert_header(project, "llk_pack_shifted.h");
    } else {
       insert_header(project, "llk_pack.h");
    }

    remove_header(project, "compute_hlk_api.h");

    vector<string> func_calls_to_remove = {
        "hlk_wait_tiles",
        "hlk_pop_tiles",
        "hlk_reconfig_unpacker_df",
    };

    replace_call_statements(project, {"hlk_get_tile"     , "hlk_release_tile"     , "hlk_debug_dump",       "hlk_reconfig_packer_df",        "hlk_flush_tiles", "hlk_relu_config",      "hlk_get_next_op_info"},
                                     {"llk_pack_get_tile", "llk_pack_release_tile", "llk_pack_debug_dump",  "llk_pack_reconfig_data_format", "llk_free_tiles",  "llk_pack_relu_config", "llk_get_next_op_info"});
    vector<string> llk_init_list;
    vector<string> hw_func_names;

    // Replaces hlk with llk statements
    if (hlk_ops.size() > 0) {
        replace_call_statements_for_pack_and_get_operand_ids(project, hlk_sfpu_op, untilize_output, hlk_matrix_op, pack_microblocks, device_name, fp32_dest_acc_en, pack_l1_acc_en);
    }

    patch_main_decl_for_llk(project, "pack_main");

    // TODO: These calls will likely fail with real multiop
    if (hlk_ops.size() > 0) {
        insert_void_argless_func_calls_after_func_call(project, "hlk_main", "hlk_acquire_dst", {hlk_matrix_op->get_pack_wait_for_math_done()});
        insert_void_argless_func_calls_after_func_call(project, "hlk_main", "hlk_release_dst", {hlk_matrix_op->get_pack_dest_section_done()});
    }
    remove_call_statements(project, {"hlk_acquire_dst", "hlk_release_dst"});

    if (hlk_ops.size() > 0) {
        insert_outer_loop(project, "hlk_main", perf_dump_en, 2); // FIXME: should be using unpack_main, but function "unpack_main" can't be found
    }

    // pack shifted is always done in col major face layout
    string llk_pack_dest_init;
    if (hlk_ops.size() > 0) {
        if(has_pack_shifted){
            llk_pack_dest_init = "llk_init_packer_dest_offset_registers<" + dst_mode_str + "," + "DstTileFaceLayout::ColMajor" + ">" ;
        }else{
            if(untilize_output){
                llk_pack_dest_init = "llk_pack_dest_init<" + dst_mode_str + ", " + dst_tile_face_layout_str + ", true>";
                if(adv_features_en){
                    const string FP32_DEST_ACC_EN_STR = fp32_dest_acc_en ? "true" : "false";
                    llk_pack_dest_init.pop_back();
                    llk_pack_dest_init += ", " + FP32_DEST_ACC_EN_STR + ">";
                }
            }else{
                llk_pack_dest_init = hlk_matrix_op->get_pack_dest_init();
            }
        }
    }

    if (std::getenv("HACK_FOR_GRAPH_INTERPRETER") != nullptr) {
        for (auto& op : hlk_ops) {
            replace_hlk_inits_and_insert_hw_configure(project, op, "pack", dst_mode_str, relu_config, untilize_output);
        }
        string hw_func_name = "llk_pack_hw_configure_disaggregated<false>";
        SgNodeContainer func_call_list = find_function_calls(project);
        SgFunctionDeclaration* main_decl = find_function_declaration(project, "hlk_main");
        SgFunctionDefinition* main_def = main_decl->get_definition();
        SgBasicBlock* body = main_def->get_body();


        SgExprListExp* expr_list = new SgExprListExp(SOURCE_POSITION);
        expr_list->append_expression(
            buildIntVal(16)
        );

        SgExprStatement* new_call_statement = build_void_func_call_statement(hw_func_name, expr_list, main_def);
        insertStatementAfter(*(body->get_statements().begin()), new_call_statement);

    } else if (multi_op) {
        for (auto& op : hlk_ops) {
            replace_hlk_inits_and_insert_hw_configure(project, op, "pack", dst_mode_str, relu_config, untilize_output);
        }
    } else if (hlk_ops.size() > 0) {
        llk_init_list = {llk_pack_init_func};
        const string pack_shift_hw_configure_str = untilize_output ? "llk_pack_shifted_hw_configure<true>" : "llk_pack_shifted_hw_configure<false>";
        hw_func_names = {insert_disaggregated_into_hw_configure(has_pack_shifted ? pack_shift_hw_configure_str : hlk_ops[0]->get_pack_hw_configure_func_str(), relu_config)};
    }

    for (auto& op : hlk_ops) {
        func_calls_to_remove.push_back(op->get_op_str());
        if (not multi_op) {
            func_calls_to_remove.push_back(op->get_op_str() + "_init");
            func_calls_to_remove.push_back(op->get_op_str() + "_init_short");
            func_calls_to_remove.push_back(op->get_op_str() + "_init_once");
        }
    }

    for (auto hlk_sfpu_op: hlk_sfpu_ops)  {
        func_calls_to_remove.push_back(hlk_sfpu_op->get_op_str());
        func_calls_to_remove.push_back(hlk_sfpu_op->get_op_str() + "_init");
    }
    remove_call_statements(project, func_calls_to_remove);

    if (hlk_ops.size() > 0) {
        insert_void_argless_func_call_to_statement_list(project, "hlk_main", llk_pack_dest_init, 1);
        insert_void_argless_func_call_to_statement_list(project, "hlk_main", "llk_setup_outputs", 1);
    }

    if (!has_pack_shifted and hlk_ops.size() > 0) {  // pack_shifted has expclicit pack_init and was replaced above
       insert_void_argless_func_call_to_statement_list(project, "hlk_main", llk_pack_init_func, 1);
    }

    if (not multi_op and hlk_ops.size() > 0) {
        cout << "HELLO: " << hw_func_names.at(0) << endl;
        insert_hw_configure_after_init(
            project,
            "hlk_main",
            hw_func_names,
            llk_init_list,
            {{16}},
            {}
        );
    }

    insert_namespace(project, "hlk_main", "NAMESPACE");
}

LLKTarget target_enum_from_string(string target_str) {
    if (target_str.compare("unpack") == 0) {
        return LLKTarget::UNPACK;
    } else if (target_str.compare("math") == 0) {
        return LLKTarget::MATH;
    } else if (target_str.compare("pack") == 0) {
        return LLKTarget::PACK;
    } else if (target_str.compare("struct_init_gen") == 0) {
        return LLKTarget::STRUCT_INIT_GEN;
    } else {
        std::cout << "Unsupported target! " << target_str << std::endl;
        throw;
    }
}

string target_string_from_enum(LLKTarget target_enum) {
    switch (target_enum) {
        case LLKTarget::UNPACK:
            return "unpack";
        case LLKTarget::MATH:
            return "math";
        case LLKTarget::PACK:
            return "pack";
        case LLKTarget::STRUCT_INIT_GEN:
            return "struct_init_gen";
        default:
            std::cout << "Unsupported target! " << target_enum << std::endl;
            throw;
    }
}

// Moving out of main() so that argument handling is contained
// Caching depends on some input arguments, currently a new cache entry is made based on values of:
//  - perf_dump_en
//  - untilize_output
//  - pack_microblocks
//  - fp32_dest_acc_en
//  - relu_config
//  - device_name
inline CompilationContext parse_arguments(int argc, char* argv[]) {
    Rose_STL_Container<string> arg_vec = CommandlineProcessing::generateArgListFromArgcArgv(argc, argv); printf ("l.size() = %zu \n",arg_vec.size());
    printf ("Preprocessor (before): argv = \n%s \n",StringUtility::listToString(arg_vec).c_str());

    // Add a test for HLKC target (unpack, math, pack)
    string llk_target = "unpack";
    if (CommandlineProcessing::isOptionWithParameter(arg_vec, "-hlkc:", "llk_target", llk_target, true)) {
        printf ("Turning on HLKC's llk_target = %s\n", llk_target.c_str());
    } else {
        printf ("Defaulting to llk_target = %s\n", llk_target.c_str());
    }

    // main output file for unpack, math, pack, struct_init_gen
    string output_file_name = "hlkc_output";
    if (CommandlineProcessing::isOptionWithParameter(arg_vec, "-rose:", "o", output_file_name, false)) {
        printf ("Output file set to = %s\n", output_file_name.c_str());
    } else {
        printf ("Defaulting to output_file = %s\n", output_file_name.c_str());
        // TODO: handle adding "-rose:o <output_file_name>" manually, low priority, we currently always supply the filename
    }

    fs::path output_dir = fs::path(output_file_name).parent_path();

    // an additional output file for unpack and pack (_llk_args.h)
    string llk_args_file_name = "";
    if (llk_target != "struct_init_gen") {
        if (CommandlineProcessing::isOptionWithParameter(arg_vec, "-hlkc:", "llk_args", llk_args_file_name, true)) {
            printf ("Output file set to = %s\n", llk_args_file_name.c_str());
        } else {
            fs::path llk_args_file = output_dir / fs::path(llk_target + "_llk_args.h");
            llk_args_file_name = llk_args_file.string();
            printf ("Defaulting to output_file = %s\n", llk_args_file_name.c_str());
        }
    }

    bool enable_cache = true;
    if (CommandlineProcessing::isOption(arg_vec, "-cache:", "on", true)) {
        printf ("Turning on HLKC's caching mechanism\n");
        enable_cache = true;
    } else if (CommandlineProcessing::isOption(arg_vec, "-cache:", "off", true)) {
        printf ("Turning off HLKC's caching mechanism\n");
        enable_cache = false;
    }

    bool perf_dump_en = false;
    if (CommandlineProcessing::isOption(arg_vec, "-perf_dump:", "1", true)) {
        printf ("Perf-Dump is enabled in hlkc.\n");
        perf_dump_en = true;
    } else if (CommandlineProcessing::isOption(arg_vec, "-perf_dump:", "0", true)) {
        printf ("Perf-Dump is disabled in hlkc\n");
        perf_dump_en = false;
    }

    bool untilize_output = false;
    if (CommandlineProcessing::isOption(arg_vec, "-untilize_output:", "1", true)) {
        printf ("Untilize op is enabled in hlkc.\n");
        untilize_output = true;
    } else if (CommandlineProcessing::isOption(arg_vec, "-untilize_output:", "0", true)) {
        printf ("Untilize op is disabled in hlkc\n");
        untilize_output = false;
    }

    bool pack_microblocks = false;
    if (CommandlineProcessing::isOption(arg_vec, "-pack_microblocks:", "1", true)) {
        printf ("Pack microblocks is enabled in hlkc.\n");
        pack_microblocks = true;
    } else if (CommandlineProcessing::isOption(arg_vec, "-pack_microblocks:", "0", true)) {
        printf ("Pack microblocks is disabled in hlkc\n");
        pack_microblocks = false;
    }

    bool fp32_dest_acc_en = false;
    if (CommandlineProcessing::isOption(arg_vec, "-fp32_dest_acc_en:", "1", true)) {
        printf ("FP32 dest is enabled in hlkc.\n");
        fp32_dest_acc_en = true;
    } else if (CommandlineProcessing::isOption(arg_vec, "-fp32_dest_acc_en:", "0", true)) {
        printf ("FP32 dest is disabled in hlkc\n");
        fp32_dest_acc_en = false;
    }

    int relu_config = 0;
    if (CommandlineProcessing::isOptionWithParameter(arg_vec, "-relu:", "config", relu_config, true)) {
        if (static_cast<uint32_t>(relu_config)>0) {
           printf ("RELU is enabled in hlkc.\n");
        } else {
           printf ("RELU is disabled in hlkc.\n");

        }
    }

    bool pack_l1_acc_en = false;
    if (CommandlineProcessing::isOption(arg_vec, "-pack_l1_acc_en:", "1", true)) {
        printf ("Pack L1 accumulation is enabled in hlkc.\n");
        pack_l1_acc_en = true;
    } else if (CommandlineProcessing::isOption(arg_vec, "-pack_l1_acc_en:", "0", true)) {
        printf ("Pack L1 accumulation is disabled in hlkc\n");
        pack_l1_acc_en = false;
    }


    string device_name = "grayskull";
    if (CommandlineProcessing::isOption(arg_vec, "-device_name:", "grayskull", true)) {
        printf ("FP32 dest is enabled in hlkc.\n");
        device_name = "grayskull";
    } else if (CommandlineProcessing::isOption(arg_vec, "-device_name:", "wormhole", true)) {
        printf ("FP32 dest is disabled in hlkc\n");
        device_name = "wormhole";
    } else if (CommandlineProcessing::isOption(arg_vec, "-device_name:", "wormhole_b0", true)) {
        printf ("FP32 dest is disabled in hlkc\n");
        device_name = "wormhole_b0";
    }


    // now arg_vec: <hlkc executable> <hlk file path> <-rose:o> <output file path>
    CompilationContext compilation_context = {.hlkc_path = fs::path(arg_vec[0]),
        .hlk_file_name = fs::path(arg_vec[1]),
        .llk_args_file_name = fs::path(llk_args_file_name).filename(),
        .llk_target = target_enum_from_string(llk_target),
        .output_dir = output_dir,
        .output_file_name = fs::path(output_file_name).filename(),
        .arg_vec = arg_vec,
        .device_name = device_name,
        .enable_cache = enable_cache,
        .perf_dump_en = perf_dump_en,
        .untilize_output = untilize_output,
        .pack_microblocks = pack_microblocks,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .relu_config = static_cast<uint32_t>(relu_config),
        .pack_l1_acc_en = pack_l1_acc_en,
    };

    return compilation_context;
}

int main(int argc, char* argv[]) {
    // Initialize and check compatibility
    ROSE_INITIALIZE;

    CompilationContext compilation_context = parse_arguments(argc, argv);

    printf ("arg_vec.size() = %zu \n", compilation_context.arg_vec.size());
    printf ("Preprocessor (after): argv = \n%s \n",StringUtility::listToString(compilation_context.arg_vec).c_str());

    // Do not use hlkc_cache in this project (for now)

    // Build the AST used by ROSE
    // enable constant folding, it will replace enum references in the AST with the compile-time constants
    // we require compile-time contants as args on several LLK functions (some checks already exist)
    SgProject* project = frontend(compilation_context.arg_vec, true);
    ROSE_ASSERT(project != NULL);

    // these could be converted to compile-time constexpr static_assert's
    HlkMatrixOp::consistency_check();
    HlkSfpuOp::consistency_check();

    if (compilation_context.llk_target != LLKTarget::STRUCT_INIT_GEN) {
        AstTests::runAllTests(project);
        //generateDOT(*project); // for debug
        check_hlk_main_exists(project);
    }

    cout << "Compilation Cache Miss! Perform compilation" << endl;
    bool adv_features_en = DeviceEnum::GRAYSKULL != device_enum_from_string(compilation_context.device_name);
    if (compilation_context.llk_target == LLKTarget::UNPACK) {
        cout << "llk_args_file_name " << compilation_context.llk_args_file_name << endl;
        generate_unpack(project, compilation_context.llk_args_file_name, compilation_context.perf_dump_en, compilation_context.device_name, compilation_context.fp32_dest_acc_en);
        dead_for_loop_elimination(project);
    } else if (compilation_context.llk_target == LLKTarget::MATH) {
        generate_math(project, compilation_context.llk_args_file_name, compilation_context.perf_dump_en, compilation_context.device_name, compilation_context.fp32_dest_acc_en);
        dead_for_loop_elimination(project);
    } else if (compilation_context.llk_target == LLKTarget::PACK) {
        cout << "llk_args_file_name " << compilation_context.llk_args_file_name << endl;
        generate_pack(project, compilation_context.llk_args_file_name, compilation_context.perf_dump_en, compilation_context.untilize_output, compilation_context.pack_microblocks, compilation_context.device_name, compilation_context.fp32_dest_acc_en, compilation_context.relu_config, compilation_context.pack_l1_acc_en);
        dead_for_loop_elimination(project);
    } else if (compilation_context.llk_target == LLKTarget::STRUCT_INIT_GEN) {
        generate_hlk_args_struct_init_generator(project, compilation_context.output_dir / compilation_context.output_file_name);
    } else {
        cout << "llk_target = " << compilation_context.llk_target << " is not supported." << endl;
        exit(1);
    }

    //generatePDF(*project);
    //generate_cfg_hlk_main(project);

    // Generate source code from AST
    if (compilation_context.llk_target != LLKTarget::STRUCT_INIT_GEN) {
        // experimenting with these tranforms -- they don't seem to help at the moment
        //legacy::PRE::partialRedundancyElimination(project);
        //cleanupInlinedCode(project); // would be required if we need to inline
        //removeUnusedVariables(project); // to be seen if useful

        AstTests::runAllTests(project);
        // generateDOT(*project); // for debug
        project->unparse();
    }

    return 0;
}
