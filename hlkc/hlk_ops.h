#include "rose.h"
#include <string>
#include "staticSingleAssignment.h"
#include "hlkc_cache.hpp"

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

class HlkOp {
    public:
        virtual string get_op_str() const = 0;
        virtual string get_math_func_str() = 0;
        virtual string get_math_init_func_str() = 0;
        virtual vector<int> get_math_func_hlk_api_args_positions_to_keep() const = 0;
        virtual vector<int> get_math_func_hlk_api_option_positions() const = 0;
};

class HlkMatrixOp : public HlkOp {
    public:
     static constexpr int NUM_HLK_MATRIX_OP_APIS = 21;

     enum OpApi : int {
         hlk_matrix_op_UNINITIALIZED = -1,
         hlk_copy_tile_to_dst = 0,
         hlk_tilize_and_copy_to_dst,
         hlk_untilize_and_copy_to_dst,
         hlk_mm_tile,
         hlk_add_tile,
         hlk_add_tile_bcast,
         hlk_add_tile_from_dst,
         hlk_add_tile_from_dst_bcast,
         hlk_multiply_tile,
         hlk_multiply_tile_bcast,
         hlk_multiply_tile_from_dst,
         hlk_multiply_tile_from_dst_bcast,
         hlk_subtract_tile,
         hlk_subtract_tile_bcast,
         hlk_subtract_tile_from_dst,
         hlk_subtract_tile_from_dst_bcast,
         hlk_reduce_tile,
         hlk_transpose_xy_tile,
         hlk_load_mm_partial_to_dst,
         hlk_broadcast_tile,
         hlk_add_tile_to_dst,
     };

     static constexpr OpApi all_op_apis[NUM_HLK_MATRIX_OP_APIS] = {
         hlk_copy_tile_to_dst,
         hlk_tilize_and_copy_to_dst,
         hlk_untilize_and_copy_to_dst,
         hlk_mm_tile,
         hlk_add_tile,
         hlk_add_tile_bcast,
         hlk_add_tile_from_dst,
         hlk_add_tile_from_dst_bcast,
         hlk_multiply_tile,
         hlk_multiply_tile_bcast,
         hlk_multiply_tile_from_dst,
         hlk_multiply_tile_from_dst_bcast,
         hlk_subtract_tile,
         hlk_subtract_tile_bcast,
         hlk_subtract_tile_from_dst,
         hlk_subtract_tile_from_dst_bcast,
         hlk_reduce_tile,
         hlk_transpose_xy_tile,
         hlk_load_mm_partial_to_dst,
         hlk_broadcast_tile,
         hlk_add_tile_to_dst,
     };

     static const string all_op_api_strs[NUM_HLK_MATRIX_OP_APIS];
     static const map<int, string> hlk_bcast_enum_to_llk_enum_str;
     static const map<int, string> hlk_reduce_func_enum_to_llk_enum_str;
     static const map<int, string> hlk_reduce_dim_enum_to_llk_enum_str;

    protected:
        OpApi op_api;
        string op_str;

        // LLK unpack
        string unpack_include_header_str;

        string unpack_init_func_str;

        string unpack_hw_configure_func_str;
        vector<int> unpack_hw_configure_func_hlk_api_expr_positions;

        string unpack_func_str;
        vector<int> unpack_func_hlk_api_operand_positions;
        vector<int> unpack_func_hlk_api_args_positions_to_keep;

        string unpack_struct_t_str;

        // LLK math
        string math_include_header_str;
        string math_init_func_str;
        string math_func_str;
        vector<int> math_func_hlk_api_args_positions_to_keep;
        vector<int> math_func_hlk_api_option_positions;
        string math_struct_t_str;

        // LLK math-pack sync
        string math_wait_for_dest_available;
        string math_dest_section_done;
        string math_pack_sync_init;
        string pack_wait_for_math_done;
        string pack_dest_section_done;
        string pack_dest_init;

        // LLK pack
        string pack_hw_configure_func_str;

        HlkMatrixOp() :
            op_api(hlk_matrix_op_UNINITIALIZED),
            op_str(""),

            // LLK unpack
            unpack_include_header_str(""),

            unpack_init_func_str(""),

            unpack_hw_configure_func_str(""),
            unpack_hw_configure_func_hlk_api_expr_positions({}),

            unpack_func_str(""),
            unpack_func_hlk_api_operand_positions({}),
            unpack_func_hlk_api_args_positions_to_keep({}),

            unpack_struct_t_str(""),
            // LLK math
            math_include_header_str(""),
            math_init_func_str(""),
            math_func_str(""),
            math_func_hlk_api_args_positions_to_keep({}),
            math_func_hlk_api_option_positions({}),
            math_struct_t_str(""),
            // LLK math-pack sync
            math_wait_for_dest_available(""),
            math_dest_section_done(""),
            math_pack_sync_init(""),
            pack_wait_for_math_done(""),
            pack_dest_section_done(""),
            pack_dest_init(""),
            // LLK pack
            pack_hw_configure_func_str("llk_pack_hw_configure") {}

    public:
        virtual HlkMatrixOp::OpApi get_op_api() {
            return op_api;
        }
        virtual string get_op_str() const override {
            return op_str;
        }

        // LLK unpack
        virtual string get_unpack_include_header_str() {
            return unpack_include_header_str;
        }
        virtual string get_unpack_init_func_str() {
            return unpack_init_func_str;
        }

        virtual string get_unpack_hw_configure_func_str() {
            return unpack_hw_configure_func_str;
        };
        virtual vector<int> get_unpack_hw_configure_func_hlk_api_expr_positions() {
            return unpack_hw_configure_func_hlk_api_expr_positions;
        };

        virtual string get_unpack_func_str() {
            return unpack_func_str;
        }
        virtual vector<int> get_unpack_func_hlk_api_operand_positions() {
            return unpack_func_hlk_api_operand_positions;
        }
        virtual vector<int> get_unpack_func_hlk_api_args_positions_to_keep() {
            return unpack_func_hlk_api_args_positions_to_keep;
        }

        virtual string get_unpack_struct_t_str() {
            return unpack_struct_t_str;
        }

        // LLK math
        virtual string get_math_include_header_str() {
            return math_include_header_str;
        }
        virtual string get_math_init_func_str() {
            return math_init_func_str;
        }
        virtual string get_math_func_str() override {
            return math_func_str;
        }
        virtual vector<int> get_math_func_hlk_api_args_positions_to_keep() const override {
            return math_func_hlk_api_args_positions_to_keep;
        }
        virtual vector<int> get_math_func_hlk_api_option_positions() const override {
            return math_func_hlk_api_option_positions;
        }
        virtual string get_math_struct_t_str() {
            return math_struct_t_str;
        }

        // LLK math/pack sync
        virtual string get_math_wait_for_dest_available() {
            return math_wait_for_dest_available;
        }
        virtual string get_math_dest_section_done() {
            return math_dest_section_done;
        }
        virtual string get_math_pack_sync_init() {
            return math_pack_sync_init;
        }
        virtual string get_pack_wait_for_math_done() {
            return pack_wait_for_math_done;
        }
        virtual string get_pack_dest_section_done() {
            return pack_dest_section_done;
        }
        virtual string get_pack_dest_init() {
            return pack_dest_init;
        }

        // LLK pack
        virtual string get_pack_hw_configure_func_str() {
            return pack_hw_configure_func_str;
        };
        virtual string get_optional_pack_init_func_str(string dst_mode_str, bool untilize_output) {
            return "llk_init_packer_dest_offset_registers<" + dst_mode_str + "," + "DstTileFaceLayout::RowMajor" + "," + (untilize_output ? "true" : "false") + ">";
        };

        virtual void specialize(const vector<int>& int_arg_vals, bool adv_features_en, bool fp32_dest_acc_en) = 0;
        virtual void specialize_math_pack_sync(string dst_mode_str, bool untilize_en, bool adv_features_en, bool fp32_dest_acc_en);
        virtual void specialize_math_func_str(const string& dst_mode_str, bool adv_features_en, bool fp32_dest_acc_en) {};
        virtual void specialize_unpack(const string& template_params){};
	    virtual void specialize_pack_func_str(const vector<int>& int_arg_vals, bool untilize_en, bool adv_features_en, bool fp32_dest_acc_en);
        virtual void specialize_hw_fp32(string& hw_func_name, bool adv_features_en, bool fp32_dest_acc_en);

        static HlkMatrixOp *make_hlk_matrix_op(HlkMatrixOp::OpApi op_api);
        static string get_op_str(HlkMatrixOp::OpApi op_api);
        static void consistency_check();
};

const string HlkMatrixOp::all_op_api_strs[NUM_HLK_MATRIX_OP_APIS] = {
    "hlk_copy_tile_to_dst",
    "hlk_tilize_and_copy_to_dst",
    "hlk_untilize_and_copy_to_dst",
    "hlk_mm_tile",
    "hlk_add_tile",
    "hlk_add_tile_bcast",
    "hlk_add_tile_from_dst",
    "hlk_add_tile_from_dst_bcast",
    "hlk_multiply_tile",
    "hlk_multiply_tile_bcast",
    "hlk_multiply_tile_from_dst",
    "hlk_multiply_tile_from_dst_bcast",
    "hlk_subtract_tile",
    "hlk_subtract_tile_bcast",
    "hlk_subtract_tile_from_dst",
    "hlk_subtract_tile_from_dst_bcast",
    "hlk_reduce_tile",
    "hlk_transpose_xy_tile",
    "hlk_load_mm_partial_to_dst",
    "hlk_broadcast_tile",
    "hlk_add_tile_to_dst",
};

const map<int, string> HlkMatrixOp::hlk_bcast_enum_to_llk_enum_str = {
    {0, "NONE"},
    {1, "ROW"},
    {2, "COL"},
    {4, "SCALAR"},
};

const map<int, string> HlkMatrixOp::hlk_reduce_func_enum_to_llk_enum_str = {
    {0, "SUM"},
    {1, "AVG"},
    {2, "MAX"},
};

const map<int, string> HlkMatrixOp::hlk_reduce_dim_enum_to_llk_enum_str = {
    {1, "REDUCE_COL"},
    {2, "REDUCE_ROW"},
    {4, "REDUCE_SCALAR"},
    /* not supported yet
    {1, "COLUMN"},
    {2, "GRID"},
    */
};

string HlkMatrixOp::get_op_str(HlkMatrixOp::OpApi op_api) {
    return all_op_api_strs[op_api];
}

void HlkMatrixOp::specialize_math_pack_sync(string dst_mode_str, bool untilize_en, bool adv_features_en, bool fp32_dest_acc_en) {
    math_pack_sync_init          = string("llk_math_pack_sync_init") + "<" + dst_mode_str + ">";
    specialize_hw_fp32(math_pack_sync_init, adv_features_en, fp32_dest_acc_en);
    math_wait_for_dest_available = string("llk_math_wait_for_dest_available") + "<" + dst_mode_str + ">";
    math_dest_section_done       = string("llk_math_dest_section_done") + "<" + dst_mode_str + ">";
    specialize_hw_fp32(math_dest_section_done, adv_features_en, fp32_dest_acc_en);

    const string DEST_LAYOUT = "DstTileFaceLayout::RowMajor";
    const string UNTILIZE_EN = untilize_en ? "true" : "false";

    pack_dest_init               = string("llk_pack_dest_init") + "<" + dst_mode_str + ", " + DEST_LAYOUT + ", " + UNTILIZE_EN + ">";
    specialize_hw_fp32(pack_dest_init, adv_features_en, fp32_dest_acc_en);

    pack_wait_for_math_done      = string("llk_packer_wait_for_math_done");

    pack_dest_section_done       = string("llk_pack_dest_section_done") + "<" + dst_mode_str + ">";
    specialize_hw_fp32(pack_dest_section_done, adv_features_en, fp32_dest_acc_en);
}

void HlkMatrixOp::specialize_pack_func_str(const vector<int>& int_arg_vals, bool untilize_en, bool adv_features_en, bool fp32_dest_acc_en) {

    const string UNTILIZE_EN = untilize_en ? "true" : "false";

    string suffix = "<" + UNTILIZE_EN + ">";
    pack_hw_configure_func_str   += suffix;
    specialize_hw_fp32(pack_hw_configure_func_str, adv_features_en, fp32_dest_acc_en);
}


void HlkMatrixOp::specialize_hw_fp32(string& hw_func_name, bool adv_features_en, bool fp32_dest_acc_en) {
    if(adv_features_en){
        const string FP32_DEST_ACC_EN_STR = fp32_dest_acc_en ? "true" : "false";
        assert((hw_func_name.size() != 0) && "Expect function already for unpack hw configure");
        if(hw_func_name.back() == '>'){
            hw_func_name.pop_back();
            hw_func_name += ", ";
        }else{
            size_t found_pos = hw_func_name.find("<");
            assert((found_pos == string::npos) && "Should not have template");
            hw_func_name += "<";
        }
        hw_func_name += FP32_DEST_ACC_EN_STR;
        hw_func_name += ">";
    }
}

void HlkMatrixOp::consistency_check() {
    int num_op_apis = 0;
    int num_strs = 0;

    for (auto op_api : all_op_apis) num_op_apis++;
    for (auto strs : all_op_api_strs) num_strs++;

    assert(num_op_apis > 0 && "Error: the number of HLK matrix op types must be > 0.");
    assert(num_op_apis == num_strs && "Error: the number of HLK matrix op types and corresponding strings must match.");
}

class HlkMatrixOp_hlk_copy_tile_to_dst : public HlkMatrixOp {
    public:
        HlkMatrixOp_hlk_copy_tile_to_dst(HlkMatrixOp::OpApi arg_op_api) : HlkMatrixOp() {
            assert(arg_op_api == HlkMatrixOp::OpApi::hlk_copy_tile_to_dst);

            op_api                       = arg_op_api;
            op_str                       = "hlk_copy_tile_to_dst";

            unpack_include_header_str    = "llk_unpack_A.h";
            unpack_init_func_str         = "llk_unpack_A_init<BroadcastType::NONE, false, false>";

            unpack_hw_configure_func_str = "llk_unpack_A_hw_configure<BroadcastType::NONE, false, false, false>";
            unpack_hw_configure_func_hlk_api_expr_positions = {};

            unpack_func_str              = "llk_unpack_A";
            unpack_func_hlk_api_args_positions_to_keep = {1,2};
            unpack_func_hlk_api_operand_positions   = {1};

            unpack_struct_t_str          = "llk_unpack_A_params_t";

            math_include_header_str        = "llk_math_eltwise_unary_datacopy.h";
            math_init_func_str = "llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, false>";
            math_func_str = "llk_math_eltwise_unary_datacopy";
            math_func_hlk_api_args_positions_to_keep = {3};
            math_func_hlk_api_option_positions = {};
            math_struct_t_str              = "llk_math_eltwise_unary_params_t";

        }

        virtual void specialize(const vector<int>& int_arg_vals, bool adv_features_en, bool fp32_dest_acc_en) {
            assert(int_arg_vals.size()==0 && "Error: cannot specialize the OP, didn't expect any additional int args.");

            specialize_hw_fp32(unpack_hw_configure_func_str, adv_features_en, fp32_dest_acc_en);
        }
        virtual void specialize_unpack(const string& template_params) {
            unpack_init_func_str += template_params;
            unpack_func_str += template_params;
        }
        virtual void specialize_math_func_str(const string& dst_mode_str, bool adv_features_en, bool fp32_dest_acc_en) {
          math_func_str += "<A2D, BroadcastType::NONE, " + dst_mode_str + ">";
        }
};

class HlkMatrixOp_hlk_tilize_and_copy_to_dst : public HlkMatrixOp {
    public:
        HlkMatrixOp_hlk_tilize_and_copy_to_dst(HlkMatrixOp::OpApi arg_op_api) : HlkMatrixOp() {
            assert(arg_op_api == HlkMatrixOp::OpApi::hlk_tilize_and_copy_to_dst);

            op_api                       = arg_op_api;
            op_str                       = "hlk_tilize_and_copy_to_dst";

            unpack_include_header_str    = "llk_unpack_tilize.h";
            unpack_init_func_str         = "llk_unpack_tilize_init";

            unpack_hw_configure_func_str = "llk_unpack_tilize_hw_configure";
            unpack_hw_configure_func_hlk_api_expr_positions = {4};

            unpack_func_str              = "llk_unpack_tilize";
            unpack_func_hlk_api_args_positions_to_keep = {1,2,4};
            unpack_func_hlk_api_operand_positions   = {1};

            unpack_struct_t_str          = "llk_unpack_tilize_params_t";

            math_include_header_str        = "llk_math_eltwise_unary_datacopy.h";
            math_init_func_str = "llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, false>";
            math_func_str = "llk_math_eltwise_unary_datacopy";
            math_func_hlk_api_args_positions_to_keep = {3};
            math_func_hlk_api_option_positions = {};
            math_struct_t_str              = "llk_math_eltwise_unary_params_t";

        }

        virtual void specialize(const vector<int>& int_arg_vals, bool adv_features_en, bool fp32_dest_acc_en) {
            assert(int_arg_vals.size()==0 && "Error: cannot specialize the OP, didn't expect any additional int args.");

            specialize_hw_fp32(unpack_hw_configure_func_str, adv_features_en, fp32_dest_acc_en);
        }
        virtual void specialize_unpack(const string& template_params) {
            unpack_init_func_str += template_params;
            unpack_func_str += template_params;
        }
        virtual void specialize_math_func_str(const string& dst_mode_str, bool adv_features_en, bool fp32_dest_acc_en) {
          math_func_str += "<A2D, BroadcastType::NONE, " + dst_mode_str + ">";
        }
};

class HlkMatrixOp_hlk_untilize_and_copy_to_dst : public HlkMatrixOp {
    public:
        HlkMatrixOp_hlk_untilize_and_copy_to_dst(HlkMatrixOp::OpApi arg_op_api) : HlkMatrixOp() {
            assert(arg_op_api == HlkMatrixOp::OpApi::hlk_untilize_and_copy_to_dst);

            op_api                       = arg_op_api;
            op_str                       = "hlk_untilize_and_copy_to_dst";

            unpack_include_header_str    = "llk_unpack_untilize.h";
            unpack_init_func_str         = "llk_unpack_untilize_init";

            unpack_hw_configure_func_str = "llk_unpack_untilize_hw_configure";
            unpack_hw_configure_func_hlk_api_expr_positions = {};

            unpack_func_str              = "llk_unpack_untilize";
            unpack_func_hlk_api_args_positions_to_keep = {1,2,4};
            unpack_func_hlk_api_operand_positions   = {1};

            unpack_struct_t_str          = "llk_unpack_untilize_params_t";

            math_include_header_str        = "llk_math_eltwise_unary_datacopy.h";
            math_init_func_str = "llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, false>";
            math_func_str = "llk_math_eltwise_unary_datacopy";
            math_func_hlk_api_args_positions_to_keep = {3};
            math_func_hlk_api_option_positions = {};
            math_struct_t_str              = "llk_math_eltwise_unary_params_t";

        }

        virtual void specialize(const vector<int>& int_arg_vals, bool adv_features_en, bool fp32_dest_acc_en) {
            assert(int_arg_vals.size()==0 && "Error: cannot specialize the OP, didn't expect any additional int args.");

            specialize_hw_fp32(unpack_hw_configure_func_str, adv_features_en, fp32_dest_acc_en);
        }
        virtual void specialize_unpack(const string& template_params) {
            unpack_init_func_str += template_params;
            unpack_func_str += template_params;
        }
        virtual void specialize_math_func_str(const string& dst_mode_str, bool adv_features_en, bool fp32_dest_acc_en) {
          math_func_str += "<A2D, BroadcastType::NONE, " + dst_mode_str + ">";
        }
};

class HlkMatrixOp_hlk_transpose_xy_tile : public HlkMatrixOp {
    public:
        HlkMatrixOp_hlk_transpose_xy_tile(HlkMatrixOp::OpApi arg_op_api) : HlkMatrixOp() {
            assert(arg_op_api == HlkMatrixOp::OpApi::hlk_transpose_xy_tile);

            op_api                       = arg_op_api;
            op_str                       = "hlk_transpose_xy_tile";

            unpack_include_header_str    = "llk_unpack_A.h";
            unpack_init_func_str         = "llk_unpack_A_init<BroadcastType::NONE, true, false>";

            unpack_hw_configure_func_str = "llk_unpack_A_hw_configure<BroadcastType::NONE, true, true, false>";
            unpack_hw_configure_func_hlk_api_expr_positions = {};

            unpack_func_str              = "llk_unpack_A<BroadcastType::NONE, true>";
            unpack_func_hlk_api_args_positions_to_keep = {1,2};
            unpack_func_hlk_api_operand_positions   = {1};

            unpack_struct_t_str          = "llk_unpack_A_params_t";

            math_include_header_str        = "llk_math_eltwise_unary_datacopy.h";
            math_init_func_str             = "llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, true>";
            math_func_str = "llk_math_eltwise_unary_datacopy";
            math_func_hlk_api_args_positions_to_keep = {3};
            math_func_hlk_api_option_positions = {};
            math_struct_t_str              = "llk_math_eltwise_unary_params_t";

        }

        virtual void specialize(const vector<int>& int_arg_vals, bool adv_features_en, bool fp32_dest_acc_en) {
            assert(int_arg_vals.size()==0 && "Error: cannot specialize the OP, didn't expect any additional int args.");
            specialize_hw_fp32(unpack_hw_configure_func_str, adv_features_en, fp32_dest_acc_en);
        }

        virtual void specialize_math_func_str(const string& dst_mode_str, bool adv_features_en, bool fp32_dest_acc_en) {
          math_func_str += "<A2D, BroadcastType::NONE, " + dst_mode_str + ">";
        }
};

class HlkMatrixOp_hlk_load_mm_partial_to_dst : public HlkMatrixOp {
   public:
    HlkMatrixOp_hlk_load_mm_partial_to_dst(HlkMatrixOp::OpApi arg_op_api) : HlkMatrixOp() {
        assert(arg_op_api == HlkMatrixOp::OpApi::hlk_load_mm_partial_to_dst);

        op_api = arg_op_api;
        op_str = "hlk_load_mm_partial_to_dst";

        unpack_include_header_str = "llk_unpack_A.h";
        unpack_init_func_str = "llk_unpack_A_init<BroadcastType::NONE, true, false>";

        unpack_hw_configure_func_str = "llk_unpack_A_hw_configure<BroadcastType::NONE, true, false, false>";
        unpack_hw_configure_func_hlk_api_expr_positions = {};

        unpack_func_str = "llk_unpack_A<BroadcastType::NONE, true>";
        unpack_func_hlk_api_args_positions_to_keep = {1, 2};
        unpack_func_hlk_api_operand_positions = {1};

        unpack_struct_t_str = "llk_unpack_A_params_t";

        math_include_header_str = "llk_math_eltwise_unary_datacopy.h";
        math_init_func_str = "llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, false>";
        math_func_str = "llk_math_eltwise_unary_datacopy";
        math_func_hlk_api_args_positions_to_keep = {3};
        math_func_hlk_api_option_positions = {};
        math_struct_t_str = "llk_math_eltwise_unary_params_t";

    }

    virtual void specialize(const vector<int>& int_arg_vals, bool adv_features_en, bool fp32_dest_acc_en) {
        assert(int_arg_vals.size() == 0 && "Error: cannot specialize the OP, didn't expect any additional int args.");
        specialize_hw_fp32(unpack_hw_configure_func_str, adv_features_en, fp32_dest_acc_en);
    }
    virtual void specialize_math_func_str(const string& dst_mode_str, bool adv_features_en, bool fp32_dest_acc_en) {
        math_func_str += "<A2D, BroadcastType::NONE, " + dst_mode_str + ">";
    }
};

class HlkMatrixOp_hlk_broadcast_tile : public HlkMatrixOp {
   public:
    HlkMatrixOp_hlk_broadcast_tile(HlkMatrixOp::OpApi arg_op_api) : HlkMatrixOp() {
        assert(arg_op_api == HlkMatrixOp::OpApi::hlk_broadcast_tile);

        op_api = arg_op_api;
        op_str = "hlk_broadcast_tile";

        unpack_include_header_str = "llk_unpack_A.h";
        unpack_init_func_str = "llk_unpack_A_init";

        unpack_hw_configure_func_str = "llk_unpack_A_hw_configure";
        unpack_hw_configure_func_hlk_api_expr_positions = {};

        unpack_func_str = "llk_unpack_A";
        unpack_func_hlk_api_args_positions_to_keep = {2, 3};
        unpack_func_hlk_api_operand_positions = {2};

        unpack_struct_t_str = "llk_unpack_A_params_t";

        math_include_header_str = "llk_math_eltwise_unary_datacopy.h";
        math_init_func_str = "llk_math_eltwise_unary_datacopy_init";
        math_func_str = "llk_math_eltwise_unary_datacopy";
        math_func_hlk_api_args_positions_to_keep = {4};
        math_func_hlk_api_option_positions = {1};
        math_struct_t_str = "llk_math_eltwise_unary_params_t";

    }

    virtual void specialize(const vector<int>& int_arg_vals, bool adv_features_en, bool fp32_dest_acc_en = false) {
        assert(int_arg_vals.size() == 1 && "Error: cannot specialize the OP, was expecting 1 int arg val.");

        string unpack_suffix = "<BroadcastType::" + hlk_bcast_enum_to_llk_enum_str.at(int_arg_vals[0]) + ">";
        string unpack_hw_suffix = "<BroadcastType::" + hlk_bcast_enum_to_llk_enum_str.at(int_arg_vals[0]) + ", false, false, false>";
        unpack_hw_configure_func_str += unpack_hw_suffix;

        specialize_hw_fp32(unpack_hw_configure_func_str, adv_features_en, fp32_dest_acc_en);

        unpack_init_func_str += unpack_suffix;
        unpack_func_str += unpack_suffix;

        math_init_func_str += "<B2D, BroadcastType::" + hlk_bcast_enum_to_llk_enum_str.at(int_arg_vals[0]) + ">";
        math_func_str += "<B2D, BroadcastType::" + hlk_bcast_enum_to_llk_enum_str.at(int_arg_vals[0]) + ",";
    }

    virtual void specialize_math_func_str(const string& dst_mode_str, bool adv_features_en, bool fp32_dest_acc_en) { math_func_str += " " + dst_mode_str + ">"; }
};
class HlkMatrixOp_hlk_mm_tile : public HlkMatrixOp {
    public:
        HlkMatrixOp_hlk_mm_tile(HlkMatrixOp::OpApi arg_op_api) : HlkMatrixOp() {
            assert(arg_op_api == HlkMatrixOp::OpApi::hlk_mm_tile);

            op_api                      = arg_op_api;
            op_str                       = "hlk_mm_tile";

            unpack_include_header_str    = "llk_unpack_AB_matmul.h";
            unpack_init_func_str         = "llk_unpack_AB_matmul_init";

            unpack_hw_configure_func_str = "llk_unpack_AB_matmul_hw_configure";
            unpack_hw_configure_func_hlk_api_expr_positions = {6};

            unpack_func_str              = "llk_unpack_AB_matmul";
            unpack_func_hlk_api_args_positions_to_keep = {1,2,3,4};
            unpack_func_hlk_api_operand_positions   = {1,2};

            unpack_struct_t_str          = "llk_unpack_AB_matmul_params_t";

            math_include_header_str        = "llk_math_matmul.h";
            math_init_func_str             = "llk_math_matmul_init"; // TODO: look into a "proper" way of dealing with template args
            math_func_str                  = "llk_math_matmul<MATH_FIDELITY>";
            math_func_hlk_api_args_positions_to_keep = {5};
            math_func_hlk_api_option_positions = {};
            math_struct_t_str              = "llk_math_matmul_params_t";

        }

        virtual void specialize(const vector<int>& int_arg_vals, bool adv_features_en, bool fp32_dest_acc_en) {
            assert(int_arg_vals.size() == 0 && "Error: cannot specialize the OP, was expecting 0 int arg val.");
            specialize_hw_fp32(unpack_hw_configure_func_str, adv_features_en, fp32_dest_acc_en);
            string math_suffix = "<MATH_FIDELITY>";
            math_init_func_str += math_suffix;

        }
        void specialize_math_pack_sync(string dst_mode_str, bool untilize_en, bool adv_features_en, bool fp32_dest_acc_en) override {
            HlkMatrixOp::specialize_math_pack_sync(dst_mode_str, untilize_en, adv_features_en, fp32_dest_acc_en);

            const string DEST_LAYOUT = "DstTileFaceLayout::RowMajor";
            const string UNTILIZE_EN = untilize_en ? "true" : "false";

            pack_dest_init           = string("llk_pack_dest_init") + "<" + dst_mode_str + ", " + DEST_LAYOUT + ", " + UNTILIZE_EN + ">";
            specialize_hw_fp32(pack_dest_init, adv_features_en, fp32_dest_acc_en);
        }
        virtual string get_optional_pack_init_func_str(string dst_mode_str, bool untilize_output) override {
            return "llk_init_packer_dest_offset_registers<" + dst_mode_str + "," + "DstTileFaceLayout::RowMajor" + "," + (untilize_output ? "true" : "false") + ">";
        };
};

class HlkMatrixOp_hlk_add_tile : public HlkMatrixOp {
    public:
        HlkMatrixOp_hlk_add_tile(HlkMatrixOp::OpApi arg_op_api) : HlkMatrixOp() {
            assert(arg_op_api == HlkMatrixOp::OpApi::hlk_add_tile);

            op_api                      = arg_op_api;
            op_str                       = "hlk_add_tile";

            unpack_include_header_str    = "llk_unpack_AB.h";
            unpack_init_func_str         = "llk_unpack_AB_init<BroadcastType::NONE>";

            unpack_hw_configure_func_str = "llk_unpack_AB_hw_configure<BroadcastType::NONE>";
            unpack_hw_configure_func_hlk_api_expr_positions = {};

            unpack_func_str              = "llk_unpack_AB";
            unpack_func_hlk_api_args_positions_to_keep = {1,2,3,4};
            unpack_func_hlk_api_operand_positions   = {1,2};

            unpack_struct_t_str          = "llk_unpack_AB_params_t";

            math_include_header_str        = "llk_math_eltwise_binary.h";
            math_init_func_str             = "llk_math_eltwise_binary_init<ELWADD, NONE>"; // TODO: look into a "proper" way of dealing with template args
            math_func_str                  = "llk_math_eltwise_binary";
            math_func_hlk_api_args_positions_to_keep = {5};
            math_func_hlk_api_option_positions = {};
            math_struct_t_str              = "llk_math_eltwise_binary_params_t";

        }

        virtual void specialize(const vector<int>& int_arg_vals, bool adv_features_en, bool fp32_dest_acc_en) {
            assert(int_arg_vals.size()==0 && "Error: cannot specialize the OP, didn't expect any additional int args.");

            specialize_hw_fp32(unpack_hw_configure_func_str, adv_features_en, fp32_dest_acc_en);
        }

        virtual void specialize_math_func_str(const string& dst_mode_str, bool adv_features_en, bool fp32_dest_acc_en) {
            const string ACC_TO_DEST = "false";
            math_func_str += "<ELWADD, NONE, " + dst_mode_str + ", MATH_FIDELITY, " + ACC_TO_DEST + ">";
            specialize_hw_fp32(math_func_str, adv_features_en, fp32_dest_acc_en);
        }

        virtual string get_optional_pack_init_func_str(string dst_mode_str, bool untilize_output) override {
            return "llk_init_packer_dest_offset_registers<" + dst_mode_str + "," + "DstTileFaceLayout::RowMajor" + "," + (untilize_output ? "true" : "false") + ">";
        };
};

class HlkMatrixOp_hlk_subtract_tile : public HlkMatrixOp {
    public:
        HlkMatrixOp_hlk_subtract_tile(HlkMatrixOp::OpApi arg_op_api) : HlkMatrixOp() {
            assert(arg_op_api == HlkMatrixOp::OpApi::hlk_subtract_tile);

            op_api                      = arg_op_api;
            op_str                       = "hlk_subtract_tile";

            unpack_include_header_str    = "llk_unpack_AB.h";
            unpack_init_func_str         = "llk_unpack_AB_init";

            unpack_hw_configure_func_str = "llk_unpack_AB_hw_configure<BroadcastType::NONE>";
            unpack_hw_configure_func_hlk_api_expr_positions = {};

            unpack_func_str              = "llk_unpack_AB";
            unpack_func_hlk_api_args_positions_to_keep = {1,2,3,4};
            unpack_func_hlk_api_operand_positions   = {1,2};

            unpack_struct_t_str          = "llk_unpack_AB_params_t";

            math_include_header_str        = "llk_math_eltwise_binary.h";
            math_init_func_str             = "llk_math_eltwise_binary_init<ELWSUB, NONE>"; // TODO: look into a "proper" way of dealing with template args
            math_func_str                  = "llk_math_eltwise_binary";
            math_func_hlk_api_args_positions_to_keep = {5};
            math_func_hlk_api_option_positions = {};
            math_struct_t_str              = "llk_math_eltwise_binary_params_t";

        }

        virtual void specialize(const vector<int>& int_arg_vals, bool adv_features_en, bool fp32_dest_acc_en = false) {
            assert(int_arg_vals.size()==0 && "Error: cannot specialize the OP, didn't expect any additional int args.");
            specialize_hw_fp32(unpack_hw_configure_func_str, adv_features_en, fp32_dest_acc_en);
        }

        virtual void specialize_math_func_str(const string& dst_mode_str, bool adv_features_en, bool fp32_dest_acc_en) {
            const string ACC_TO_DEST = "false";
            math_func_str += "<ELWSUB, NONE, " + dst_mode_str + ", MATH_FIDELITY, " + ACC_TO_DEST + ">";
            specialize_hw_fp32(math_func_str, adv_features_en, fp32_dest_acc_en);
        }

        virtual string get_optional_pack_init_func_str(string dst_mode_str, bool untilize_output) override {
            return "llk_init_packer_dest_offset_registers<" + dst_mode_str + "," + "DstTileFaceLayout::RowMajor" + "," + (untilize_output ? "true" : "false") + ">" ;
        };
};

class HlkMatrixOp_hlk_multiply_tile : public HlkMatrixOp {
    public:
        HlkMatrixOp_hlk_multiply_tile(HlkMatrixOp::OpApi arg_op_api) : HlkMatrixOp() {
            assert(arg_op_api == HlkMatrixOp::OpApi::hlk_multiply_tile);

            op_api                      = arg_op_api;
            op_str                       = "hlk_multiply_tile";

            unpack_include_header_str    = "llk_unpack_AB.h";
            unpack_init_func_str         = "llk_unpack_AB_init<BroadcastType::NONE>";

            unpack_hw_configure_func_str = "llk_unpack_AB_hw_configure<BroadcastType::NONE>";
            unpack_hw_configure_func_hlk_api_expr_positions = {};

            unpack_func_str              = "llk_unpack_AB";
            unpack_func_hlk_api_args_positions_to_keep = {1,2,3,4};
            unpack_func_hlk_api_operand_positions   = {1,2};

            unpack_struct_t_str          = "llk_unpack_AB_params_t";

            math_include_header_str        = "llk_math_eltwise_binary.h";
            math_init_func_str             = "llk_math_eltwise_binary_init<ELWMUL, NONE, MATH_FIDELITY>"; // TODO: look into a "proper" way of dealing with template args
            math_func_str                  = "llk_math_eltwise_binary";
            math_func_hlk_api_args_positions_to_keep = {5};
            math_func_hlk_api_option_positions = {};
            math_struct_t_str              = "llk_math_eltwise_binary_params_t";

        }

        virtual void specialize(const vector<int>& int_arg_vals, bool adv_features_en, bool fp32_dest_acc_en) {
            assert(int_arg_vals.size()==0 && "Error: cannot specialize the OP, didn't expect any additional int args.");
            specialize_hw_fp32(unpack_hw_configure_func_str, adv_features_en, fp32_dest_acc_en);
        }

        virtual void specialize_math_func_str(const string& dst_mode_str, bool adv_features_en, bool fp32_dest_acc_en) {
            const string ACC_TO_DEST = "false";
            math_func_str += "<ELWMUL, NONE, " + dst_mode_str + ", MATH_FIDELITY, " + ACC_TO_DEST + ">";
            specialize_hw_fp32(math_func_str, adv_features_en, fp32_dest_acc_en);
        }
};

class HlkMatrixOp_hlk_add_tile_bcast : public HlkMatrixOp {
    public:
        HlkMatrixOp_hlk_add_tile_bcast(HlkMatrixOp::OpApi arg_op_api) : HlkMatrixOp() {
            assert(arg_op_api == HlkMatrixOp::OpApi::hlk_add_tile_bcast);

            op_api                      = arg_op_api;
            op_str                       = "hlk_add_tile_bcast";

            unpack_include_header_str    = "llk_unpack_AB.h";
            unpack_init_func_str         = "llk_unpack_AB_init";

            unpack_hw_configure_func_str = "llk_unpack_AB_hw_configure";
            unpack_hw_configure_func_hlk_api_expr_positions = {};

            unpack_func_str              = "llk_unpack_AB";
            unpack_func_hlk_api_args_positions_to_keep = {2,3,4,5};
            unpack_func_hlk_api_operand_positions   = {2,3};

            unpack_struct_t_str          = "llk_unpack_AB_params_t";

            math_include_header_str        = "llk_math_eltwise_binary.h";
            math_init_func_str             = "llk_math_eltwise_binary_init";
            math_func_str                  = "llk_math_eltwise_binary";
            math_func_hlk_api_args_positions_to_keep = {6};
            math_func_hlk_api_option_positions = {1};
            math_struct_t_str              = "llk_math_eltwise_binary_params_t";

        }

        virtual void specialize(const vector<int>& int_arg_vals, bool adv_features_en, bool fp32_dest_acc_en = false) {
            assert(int_arg_vals.size()==1 && "Error: cannot specialize the OP, was expecting 1 int arg val.");

            string unpack_suffix = "<BroadcastType::" + hlk_bcast_enum_to_llk_enum_str.at(int_arg_vals[0]) + ">";
            unpack_hw_configure_func_str += unpack_suffix;

            specialize_hw_fp32(unpack_hw_configure_func_str, adv_features_en, fp32_dest_acc_en);

            unpack_init_func_str += unpack_suffix;
            unpack_func_str += unpack_suffix;

            math_init_func_str += "<ELWADD, BroadcastType::" + hlk_bcast_enum_to_llk_enum_str.at(int_arg_vals[0]) + ">";
            math_func_str += "<ELWADD, BroadcastType::" + hlk_bcast_enum_to_llk_enum_str.at(int_arg_vals[0]) + ",";
        }

        virtual void specialize_math_func_str(const string& dst_mode_str, bool adv_features_en, bool fp32_dest_acc_en) {
            const string ACC_TO_DEST = "false";
            math_func_str += " " + dst_mode_str + ", MATH_FIDELITY, " + ACC_TO_DEST + ">";
            specialize_hw_fp32(math_func_str, adv_features_en, fp32_dest_acc_en);
        }
};

class HlkMatrixOp_hlk_multiply_tile_bcast : public HlkMatrixOp {
    public:
        HlkMatrixOp_hlk_multiply_tile_bcast(HlkMatrixOp::OpApi arg_op_api) : HlkMatrixOp() {
            assert(arg_op_api == HlkMatrixOp::OpApi::hlk_multiply_tile_bcast);

            op_api                      = arg_op_api;
            op_str                       = "hlk_multiply_tile_bcast";

            unpack_include_header_str    = "llk_unpack_AB.h";
            unpack_init_func_str         = "llk_unpack_AB_init";

            unpack_hw_configure_func_str = "llk_unpack_AB_hw_configure";
            unpack_hw_configure_func_hlk_api_expr_positions = {};

            unpack_func_str              = "llk_unpack_AB";
            unpack_func_hlk_api_args_positions_to_keep = {2,3,4,5};
            unpack_func_hlk_api_operand_positions   = {2,3};

            unpack_struct_t_str          = "llk_unpack_AB_params_t";

            math_include_header_str        = "llk_math_eltwise_binary.h";
            math_init_func_str             = "llk_math_eltwise_binary_init";
            math_func_str                  = "llk_math_eltwise_binary";
            math_func_hlk_api_args_positions_to_keep = {6};
            math_func_hlk_api_option_positions = {1};
            math_struct_t_str              = "llk_math_eltwise_binary_params_t";
        }

        virtual void specialize(const vector<int>& int_arg_vals, bool adv_features_en, bool fp32_dest_acc_en = false) {
            assert(int_arg_vals.size()==1 && "Error: cannot specialize the OP, was expecting 1 int arg val.");

            string unpack_suffix = "<BroadcastType::" + hlk_bcast_enum_to_llk_enum_str.at(int_arg_vals[0]) + ">";
            unpack_hw_configure_func_str += unpack_suffix;

            specialize_hw_fp32(unpack_hw_configure_func_str, adv_features_en, fp32_dest_acc_en);

            unpack_init_func_str += unpack_suffix;
            unpack_func_str += unpack_suffix;

            math_init_func_str += "<ELWMUL, BroadcastType::" + hlk_bcast_enum_to_llk_enum_str.at(int_arg_vals[0]) + ", MATH_FIDELITY>";
            math_func_str += "<ELWMUL, BroadcastType::" + hlk_bcast_enum_to_llk_enum_str.at(int_arg_vals[0]) + ",";
        }

        virtual void specialize_math_func_str(const string& dst_mode_str, bool adv_features_en, bool fp32_dest_acc_en) {
            const string ACC_TO_DEST = "false";
            math_func_str += " " + dst_mode_str + ", MATH_FIDELITY, " + ACC_TO_DEST + ">";
            specialize_hw_fp32(math_func_str, adv_features_en, fp32_dest_acc_en);
        }
};

class HlkMatrixOp_hlk_subtract_tile_bcast : public HlkMatrixOp {
    public:
        HlkMatrixOp_hlk_subtract_tile_bcast(HlkMatrixOp::OpApi arg_op_api) : HlkMatrixOp() {
            assert(arg_op_api == HlkMatrixOp::OpApi::hlk_subtract_tile_bcast);

            op_api                      = arg_op_api;
            op_str                       = "hlk_subtract_tile_bcast";

            unpack_include_header_str    = "llk_unpack_AB.h";
            unpack_init_func_str         = "llk_unpack_AB_init";

            unpack_hw_configure_func_str = "llk_unpack_AB_hw_configure";
            unpack_hw_configure_func_hlk_api_expr_positions = {};

            unpack_func_str              = "llk_unpack_AB";
            unpack_func_hlk_api_args_positions_to_keep = {2,3,4,5};
            unpack_func_hlk_api_operand_positions   = {2,3};

            unpack_struct_t_str          = "llk_unpack_AB_params_t";

            math_include_header_str        = "llk_math_eltwise_binary.h";
            math_init_func_str             = "llk_math_eltwise_binary_init";
            math_func_str                  = "llk_math_eltwise_binary";
            math_func_hlk_api_args_positions_to_keep = {6};
            math_func_hlk_api_option_positions = {1};
            math_struct_t_str              = "llk_math_eltwise_binary_params_t";
        }

        virtual void specialize(const vector<int>& int_arg_vals, bool adv_features_en, bool fp32_dest_acc_en = false) {
            assert(int_arg_vals.size()==1 && "Error: cannot specialize the OP, was expecting 1 int arg val.");

            string unpack_suffix = "<BroadcastType::" + hlk_bcast_enum_to_llk_enum_str.at(int_arg_vals[0]) + ">";
            unpack_hw_configure_func_str += unpack_suffix;

            specialize_hw_fp32(unpack_hw_configure_func_str, adv_features_en, fp32_dest_acc_en);

            unpack_init_func_str += unpack_suffix;
            unpack_func_str += unpack_suffix;

            math_init_func_str += "<ELWSUB, BroadcastType::" + hlk_bcast_enum_to_llk_enum_str.at(int_arg_vals[0]) + ">";
            math_func_str += "<ELWSUB, BroadcastType::" + hlk_bcast_enum_to_llk_enum_str.at(int_arg_vals[0]) + ",";
        }

        virtual void specialize_math_func_str(const string& dst_mode_str, bool adv_features_en, bool fp32_dest_acc_en) {
            const string ACC_TO_DEST = "false";
            math_func_str += " " + dst_mode_str + ", MATH_FIDELITY, " + ACC_TO_DEST + ">";
            specialize_hw_fp32(math_func_str, adv_features_en, fp32_dest_acc_en);
        }
};

class HlkMatrixOp_hlk_add_tile_from_dst : public HlkMatrixOp {
    public:
        HlkMatrixOp_hlk_add_tile_from_dst(HlkMatrixOp::OpApi arg_op_api) : HlkMatrixOp() {
            assert(arg_op_api == HlkMatrixOp::OpApi::hlk_add_tile_from_dst);

            op_api                      = arg_op_api;
            op_str                       = "hlk_add_tile_from_dst";

            unpack_include_header_str    = "llk_unpack_A.h";
            unpack_init_func_str         = "llk_unpack_A_init<BroadcastType::NONE, false, true>";

            unpack_hw_configure_func_str = "llk_unpack_A_hw_configure";
            unpack_hw_configure_func_hlk_api_expr_positions = {};

            unpack_func_str              = "llk_unpack_A<BroadcastType::NONE, false, true>"; // Configure unpack B
            unpack_func_hlk_api_args_positions_to_keep = {1,2};
            unpack_func_hlk_api_operand_positions   = {1};

            unpack_struct_t_str          = "llk_unpack_A_params_t";

            math_include_header_str        = "llk_math_eltwise_binary.h";
            math_init_func_str             = "llk_math_eltwise_binary_init<ELWADD, NONE, 0, true>"; // TODO: look into a "proper" way of dealing with template args
            math_func_str                  = "llk_math_eltwise_binary";
            math_func_hlk_api_args_positions_to_keep = {3};
            math_func_hlk_api_option_positions = {};
            math_struct_t_str              = "llk_math_eltwise_binary_params_t";
        }

        virtual void specialize(const vector<int>& int_arg_vals, bool adv_features_en, bool fp32_dest_acc_en = false) {
            assert(int_arg_vals.size()==0 && "Error: cannot specialize the OP, didn't expect any additional int args.");
            specialize_hw_fp32(unpack_hw_configure_func_str, adv_features_en, fp32_dest_acc_en);
        }

        virtual void specialize_math_func_str(const string& dst_mode_str, bool adv_features_en, bool fp32_dest_acc_en) {
            math_func_str += "<ELWADD, NONE, " + dst_mode_str + ", 0, true>";
            specialize_hw_fp32(math_func_str, adv_features_en, fp32_dest_acc_en);
        }

        virtual string get_optional_pack_init_func_str(string dst_mode_str, bool untilize_output) override {
            return "llk_init_packer_dest_offset_registers<" + dst_mode_str + "," + "DstTileFaceLayout::RowMajor" + "," + (untilize_output ? "true" : "false") + ">" ;
        };
};

class HlkMatrixOp_hlk_subtract_tile_from_dst : public HlkMatrixOp {
    public:
        HlkMatrixOp_hlk_subtract_tile_from_dst(HlkMatrixOp::OpApi arg_op_api) : HlkMatrixOp() {
            assert(arg_op_api == HlkMatrixOp::OpApi::hlk_subtract_tile_from_dst);

            op_api                      = arg_op_api;
            op_str                       = "hlk_subtract_tile_from_dst";

            unpack_include_header_str    = "llk_unpack_A.h";
            unpack_init_func_str         = "llk_unpack_A_init<BroadcastType::NONE, false, true>";

            unpack_hw_configure_func_str = "llk_unpack_A_hw_configure";
            unpack_hw_configure_func_hlk_api_expr_positions = {};

            unpack_func_str              = "llk_unpack_A<BroadcastType::NONE, false, true>"; // Configure unpack B
            unpack_func_hlk_api_args_positions_to_keep = {1,2};
            unpack_func_hlk_api_operand_positions   = {1};

            unpack_struct_t_str          = "llk_unpack_A_params_t";

            math_include_header_str        = "llk_math_eltwise_binary.h";
            math_init_func_str             = "llk_math_eltwise_binary_init<ELWSUB, NONE, 0, true>"; // TODO: look into a "proper" way of dealing with template args
            math_func_str                  = "llk_math_eltwise_binary";
            math_func_hlk_api_args_positions_to_keep = {3};
            math_func_hlk_api_option_positions = {};
            math_struct_t_str              = "llk_math_eltwise_binary_params_t";
        }

        virtual void specialize(const vector<int>& int_arg_vals, bool adv_features_en, bool fp32_dest_acc_en = false) {
            assert(int_arg_vals.size()==0 && "Error: cannot specialize the OP, didn't expect any additional int args.");
            specialize_hw_fp32(unpack_hw_configure_func_str, adv_features_en, fp32_dest_acc_en);
        }

        virtual void specialize_math_func_str(const string& dst_mode_str, bool adv_features_en, bool fp32_dest_acc_en) {
            math_func_str += "<ELWSUB, NONE, " + dst_mode_str + ", 0, true>";
            specialize_hw_fp32(math_func_str, adv_features_en, fp32_dest_acc_en);
        }

        virtual string get_optional_pack_init_func_str(string dst_mode_str, bool untilize_output) override {
            return "llk_init_packer_dest_offset_registers<" + dst_mode_str + "," + "DstTileFaceLayout::RowMajor" + "," + (untilize_output ? "true" : "false") + ">" ;
        };
};

class HlkMatrixOp_hlk_multiply_tile_from_dst : public HlkMatrixOp {
    public:
        HlkMatrixOp_hlk_multiply_tile_from_dst(HlkMatrixOp::OpApi arg_op_api) : HlkMatrixOp() {
            assert(arg_op_api == HlkMatrixOp::OpApi::hlk_multiply_tile_from_dst);

            op_api                      = arg_op_api;
            op_str                       = "hlk_multiply_tile_from_dst";

            unpack_include_header_str    = "llk_unpack_A.h";
            unpack_init_func_str         = "llk_unpack_A_init<BroadcastType::NONE, false, true>";

            unpack_hw_configure_func_str = "llk_unpack_A_hw_configure";
            unpack_hw_configure_func_hlk_api_expr_positions = {};

            unpack_func_str              = "llk_unpack_A<BroadcastType::NONE, false, true>"; // Configure unpack B
            unpack_func_hlk_api_args_positions_to_keep = {1,2};
            unpack_func_hlk_api_operand_positions   = {1};

            unpack_struct_t_str          = "llk_unpack_A_params_t";

            math_include_header_str        = "llk_math_eltwise_binary.h";
            math_init_func_str             = "llk_math_eltwise_binary_init<ELWMUL, NONE, MATH_FIDELITY, true>"; // TODO: look into a "proper" way of dealing with template args
            math_func_str                  = "llk_math_eltwise_binary";
            math_func_hlk_api_args_positions_to_keep = {3};
            math_func_hlk_api_option_positions = {};
            math_struct_t_str              = "llk_math_eltwise_binary_params_t";
        }

        virtual void specialize(const vector<int>& int_arg_vals, bool adv_features_en, bool fp32_dest_acc_en = false) {
            assert(int_arg_vals.size()==0 && "Error: cannot specialize the OP, didn't expect any additional int args.");
            specialize_hw_fp32(unpack_hw_configure_func_str, adv_features_en, fp32_dest_acc_en);
        }

        virtual void specialize_math_func_str(const string& dst_mode_str, bool adv_features_en, bool fp32_dest_acc_en) {
            const string ACC_TO_DEST = "true";
            math_func_str += "<ELWMUL, NONE, " + dst_mode_str + ", MATH_FIDELITY, " + ACC_TO_DEST + ">";
            specialize_hw_fp32(math_func_str, adv_features_en, fp32_dest_acc_en);
        }

        virtual string get_optional_pack_init_func_str(string dst_mode_str, bool untilize_output) override {
            return "llk_init_packer_dest_offset_registers<" + dst_mode_str + "," + "DstTileFaceLayout::RowMajor" + "," + (untilize_output ? "true" : "false") + ">" ;
        };
};

class HlkMatrixOp_hlk_add_tile_from_dst_bcast : public HlkMatrixOp {
    public:
        HlkMatrixOp_hlk_add_tile_from_dst_bcast(HlkMatrixOp::OpApi arg_op_api) : HlkMatrixOp() {
            assert(arg_op_api == HlkMatrixOp::OpApi::hlk_add_tile_from_dst_bcast);

            op_api                      = arg_op_api;
            op_str                       = "hlk_add_tile_from_dst_bcast";

            unpack_include_header_str    = "llk_unpack_A.h";
            unpack_init_func_str         = "llk_unpack_A_init";

            unpack_hw_configure_func_str = "llk_unpack_A_hw_configure";
            unpack_hw_configure_func_hlk_api_expr_positions = {};

            unpack_func_str              = "llk_unpack_A";
            unpack_func_hlk_api_args_positions_to_keep = {2,3};
            unpack_func_hlk_api_operand_positions   = {2};

            unpack_struct_t_str          = "llk_unpack_A_params_t";

            math_include_header_str        = "llk_math_eltwise_binary.h";
            math_init_func_str             = "llk_math_eltwise_binary_init";
            math_func_str                  = "llk_math_eltwise_binary";
            math_func_hlk_api_args_positions_to_keep = {4};
            math_func_hlk_api_option_positions = {1};
            math_struct_t_str              = "llk_math_eltwise_binary_params_t";
        }

        virtual void specialize(const vector<int>& int_arg_vals, bool adv_features_en, bool fp32_dest_acc_en = false) {
            assert(int_arg_vals.size()==1 && "Error: cannot specialize the OP, was expecting 1 int arg val.");

            string unpack_suffix = "<BroadcastType::" + hlk_bcast_enum_to_llk_enum_str.at(int_arg_vals[0]) + ", false, false, true>";
            unpack_hw_configure_func_str += unpack_suffix;

            specialize_hw_fp32(unpack_hw_configure_func_str, adv_features_en, fp32_dest_acc_en);

            unpack_suffix = "<BroadcastType::" + hlk_bcast_enum_to_llk_enum_str.at(int_arg_vals[0]) + ", false, true>";
            unpack_init_func_str += unpack_suffix;
            unpack_func_str += unpack_suffix;

            math_init_func_str += "<ELWADD, BroadcastType::" + hlk_bcast_enum_to_llk_enum_str.at(int_arg_vals[0]) + ", 0, true>";
            math_func_str += "<ELWADD, BroadcastType::" + hlk_bcast_enum_to_llk_enum_str.at(int_arg_vals[0]) + ",";
        }

        virtual void specialize_math_func_str(const string& dst_mode_str, bool adv_features_en, bool fp32_dest_acc_en) {
            const string ACC_TO_DEST = "true";
            math_func_str += " " + dst_mode_str + ", 0, " + ACC_TO_DEST + ">";
            specialize_hw_fp32(math_func_str, adv_features_en, fp32_dest_acc_en);
        }
};

class HlkMatrixOp_hlk_multiply_tile_from_dst_bcast : public HlkMatrixOp {
    public:
        HlkMatrixOp_hlk_multiply_tile_from_dst_bcast(HlkMatrixOp::OpApi arg_op_api) : HlkMatrixOp() {
            assert(arg_op_api == HlkMatrixOp::OpApi::hlk_multiply_tile_from_dst_bcast);

            op_api                      = arg_op_api;
            op_str                       = "hlk_multiply_tile_from_dst_bcast";

            unpack_include_header_str    = "llk_unpack_A.h";
            unpack_init_func_str         = "llk_unpack_A_init";

            unpack_hw_configure_func_str = "llk_unpack_A_hw_configure";
            unpack_hw_configure_func_hlk_api_expr_positions = {};

            unpack_func_str              = "llk_unpack_A";
            unpack_func_hlk_api_args_positions_to_keep = {2,3};
            unpack_func_hlk_api_operand_positions   = {2};

            unpack_struct_t_str          = "llk_unpack_A_params_t";

            math_include_header_str        = "llk_math_eltwise_binary.h";
            math_init_func_str             = "llk_math_eltwise_binary_init";
            math_func_str                  = "llk_math_eltwise_binary";
            math_func_hlk_api_args_positions_to_keep = {4};
            math_func_hlk_api_option_positions = {1};
            math_struct_t_str              = "llk_math_eltwise_binary_params_t";
        }

        virtual void specialize(const vector<int>& int_arg_vals, bool adv_features_en, bool fp32_dest_acc_en = false) {
            assert(int_arg_vals.size()==1 && "Error: cannot specialize the OP, was expecting 1 int arg val.");

            string unpack_suffix = "<BroadcastType::" + hlk_bcast_enum_to_llk_enum_str.at(int_arg_vals[0]) + ", false, false, true>";
            unpack_hw_configure_func_str += unpack_suffix;

            specialize_hw_fp32(unpack_hw_configure_func_str, adv_features_en, fp32_dest_acc_en);

            unpack_suffix = "<BroadcastType::" + hlk_bcast_enum_to_llk_enum_str.at(int_arg_vals[0]) + ", false, true>";
            unpack_init_func_str += unpack_suffix;
            unpack_func_str += unpack_suffix;

            math_init_func_str += "<ELWMUL, BroadcastType::" + hlk_bcast_enum_to_llk_enum_str.at(int_arg_vals[0]) + ", MATH_FIDELITY, true>";
            math_func_str += "<ELWMUL, BroadcastType::" + hlk_bcast_enum_to_llk_enum_str.at(int_arg_vals[0]) + ",";
        }

        virtual void specialize_math_func_str(const string& dst_mode_str, bool adv_features_en, bool fp32_dest_acc_en) {
            const string ACC_TO_DEST = "true";
            math_func_str += " " + dst_mode_str + ", MATH_FIDELITY, " + ACC_TO_DEST + ">";
            specialize_hw_fp32(math_func_str, adv_features_en, fp32_dest_acc_en);
        }
};

class HlkMatrixOp_hlk_subtract_tile_from_dst_bcast : public HlkMatrixOp {
    public:
        HlkMatrixOp_hlk_subtract_tile_from_dst_bcast(HlkMatrixOp::OpApi arg_op_api) : HlkMatrixOp() {
            assert(arg_op_api == HlkMatrixOp::OpApi::hlk_subtract_tile_from_dst_bcast);

            op_api                      = arg_op_api;
            op_str                       = "hlk_subtract_tile_from_dst_bcast";

            unpack_include_header_str    = "llk_unpack_A.h";
            unpack_init_func_str         = "llk_unpack_A_init";

            unpack_hw_configure_func_str = "llk_unpack_A_hw_configure";
            unpack_hw_configure_func_hlk_api_expr_positions = {};

            unpack_func_str              = "llk_unpack_A";
            unpack_func_hlk_api_args_positions_to_keep = {2,3};
            unpack_func_hlk_api_operand_positions   = {2};

            unpack_struct_t_str          = "llk_unpack_A_params_t";

            math_include_header_str        = "llk_math_eltwise_binary.h";
            math_init_func_str             = "llk_math_eltwise_binary_init";
            math_func_str                  = "llk_math_eltwise_binary";
            math_func_hlk_api_args_positions_to_keep = {4};
            math_func_hlk_api_option_positions = {1};
            math_struct_t_str              = "llk_math_eltwise_binary_params_t";
        }

        virtual void specialize(const vector<int>& int_arg_vals, bool adv_features_en, bool fp32_dest_acc_en = false) {
            assert(int_arg_vals.size()==1 && "Error: cannot specialize the OP, was expecting 1 int arg val.");

            string unpack_suffix = "<BroadcastType::" + hlk_bcast_enum_to_llk_enum_str.at(int_arg_vals[0]) + ", false, false, true>";
            unpack_hw_configure_func_str += unpack_suffix;

            specialize_hw_fp32(unpack_hw_configure_func_str, adv_features_en, fp32_dest_acc_en);

            unpack_suffix = "<BroadcastType::" + hlk_bcast_enum_to_llk_enum_str.at(int_arg_vals[0]) + ", false, true>";
            unpack_init_func_str += unpack_suffix;
            unpack_func_str += unpack_suffix;

            math_init_func_str += "<ELWADD, BroadcastType::" + hlk_bcast_enum_to_llk_enum_str.at(int_arg_vals[0]) + ", 0, true>";
            math_func_str += "<ELWSUB, BroadcastType::" + hlk_bcast_enum_to_llk_enum_str.at(int_arg_vals[0]) + ",";
        }

        virtual void specialize_math_func_str(const string& dst_mode_str, bool adv_features_en, bool fp32_dest_acc_en) {
            const string ACC_TO_DEST = "true";
            math_func_str += " " + dst_mode_str + ", 0, " + ACC_TO_DEST + ">";
            specialize_hw_fp32(math_func_str, adv_features_en, fp32_dest_acc_en);
        }
};

class HlkMatrixOp_hlk_reduce_tile : public HlkMatrixOp {
    public:
        HlkMatrixOp_hlk_reduce_tile(HlkMatrixOp::OpApi arg_op_api) : HlkMatrixOp() {
            assert(arg_op_api == HlkMatrixOp::OpApi::hlk_reduce_tile);

            op_api                      = arg_op_api;
            op_str                       = "hlk_reduce_tile";

            unpack_include_header_str    = "llk_unpack_reduce.h";
            unpack_init_func_str         = "llk_unpack_reduce_init";

            unpack_hw_configure_func_str = "llk_unpack_reduce_hw_configure";
            unpack_hw_configure_func_hlk_api_expr_positions = {6};

            unpack_func_str              = "llk_unpack_reduce";
            unpack_func_hlk_api_args_positions_to_keep = {3,4};
            unpack_func_hlk_api_operand_positions   = {3};

            unpack_struct_t_str          = "llk_unpack_reduce_params_t";

            math_include_header_str        = "llk_math_reduce.h";
            math_init_func_str             = "llk_math_reduce_init";
            math_func_str                  = "llk_math_reduce";
            math_func_hlk_api_args_positions_to_keep = {5};
            math_func_hlk_api_option_positions = {1,2};
            math_struct_t_str              = "llk_math_reduce_params_t";

            pack_hw_configure_func_str = "llk_pack_reduce_hw_configure";

        }

        virtual void specialize(const vector<int>& int_arg_vals, bool adv_features_en, bool fp32_dest_acc_en = false) {
            assert(int_arg_vals.size()==2 && "Error: cannot specialize the OP, was expecting 2 int arg val.");

            string reduce_func_str = hlk_reduce_func_enum_to_llk_enum_str.at(int_arg_vals[0]);
            string dim_str = hlk_reduce_dim_enum_to_llk_enum_str.at(int_arg_vals[1]);

            string suffix_math = "<PoolType::" + reduce_func_str + "," + "ReduceDim::" + dim_str + "," + "MATH_FIDELITY" + ">";
            string suffix_unpack_pack = "<PoolType::" + reduce_func_str + "," + "ReduceDim::" + dim_str + ">";

            unpack_hw_configure_func_str += suffix_unpack_pack;

            specialize_hw_fp32(unpack_hw_configure_func_str, adv_features_en, fp32_dest_acc_en);

            unpack_init_func_str         += suffix_unpack_pack;
            unpack_func_str              += suffix_unpack_pack;
            math_init_func_str           += suffix_math;
            math_func_str                += suffix_math;
            pack_hw_configure_func_str   += suffix_unpack_pack;

            specialize_hw_fp32(math_func_str, adv_features_en, fp32_dest_acc_en);
            specialize_hw_fp32(pack_hw_configure_func_str, adv_features_en, fp32_dest_acc_en);
        }

        virtual void specialize_pack_func_str(const vector<int>& int_arg_vals, bool untilize_en, bool adv_features_en, bool fp32_dest_acc_en) {
            assert(int_arg_vals.size()==2 && "Error: cannot specialize the OP, was expecting 2 int arg val.");
            const string UNTILIZE_EN = untilize_en ? "true" : "false";

            string reduce_func_str = hlk_reduce_func_enum_to_llk_enum_str.at(int_arg_vals[0]);
            string dim_str = hlk_reduce_dim_enum_to_llk_enum_str.at(int_arg_vals[1]);

            string suffix = "<" + UNTILIZE_EN + "," + "PoolType::" + reduce_func_str + "," + "ReduceDim::" + dim_str + ">";

            pack_hw_configure_func_str   += suffix;

            specialize_hw_fp32(pack_hw_configure_func_str, adv_features_en, fp32_dest_acc_en);
        }
};

class HlkMatrixOp_hlk_add_tile_to_dst : public HlkMatrixOp {
    public:
        HlkMatrixOp_hlk_add_tile_to_dst(HlkMatrixOp::OpApi arg_op_api) : HlkMatrixOp() {
            assert(arg_op_api == HlkMatrixOp::OpApi::hlk_add_tile_to_dst);

            op_api                      = arg_op_api;
            op_str                       = "hlk_add_tile_to_dst";

            unpack_include_header_str    = "llk_unpack_A.h";
            unpack_init_func_str         = "llk_unpack_A_init<BroadcastType::NONE, false, true>";

            unpack_hw_configure_func_str = "llk_unpack_A_hw_configure<BroadcastType::NONE, false, false, true>";
            unpack_hw_configure_func_hlk_api_expr_positions = {};

            unpack_func_str              = "llk_unpack_A<BroadcastType::NONE, false, true>";
            unpack_func_hlk_api_args_positions_to_keep = {1,2};
            unpack_func_hlk_api_operand_positions   = {1};

            unpack_struct_t_str          = "llk_unpack_A_params_t";

            math_include_header_str        = "llk_math_eltwise_binary.h";
            math_init_func_str             = "llk_math_eltwise_binary_init<ELWADD, NONE, 0, true>"; // TODO: look into a "proper" way of dealing with template args
            math_func_str                  = "llk_math_eltwise_binary";
            math_func_hlk_api_args_positions_to_keep = {3,4};
            math_func_hlk_api_option_positions = {};
            math_struct_t_str              = "llk_math_eltwise_binary_params_t";

        }

        virtual void specialize(const vector<int>& int_arg_vals, bool adv_features_en, bool fp32_dest_acc_en = false) {
            assert(int_arg_vals.size()==0 && "Error: cannot specialize the OP, didn't expect any additional int args.");
            specialize_hw_fp32(unpack_hw_configure_func_str, adv_features_en, fp32_dest_acc_en);
        }

        virtual void specialize_math_func_str(const string& dst_mode_str, bool adv_features_en, bool fp32_dest_acc_en) {
            math_func_str += "<ELWADD, NONE, " + dst_mode_str + ", 0, true>";
            specialize_hw_fp32(math_func_str, adv_features_en, fp32_dest_acc_en);
        }

        virtual string get_optional_pack_init_func_str(string dst_mode_str, bool untilize_output) override {
            return "llk_init_packer_dest_offset_registers<" + dst_mode_str + "," + "DstTileFaceLayout::RowMajor" + "," + (untilize_output ? "true" : "false") + ">" ;
        };
};

HlkMatrixOp* HlkMatrixOp::make_hlk_matrix_op(HlkMatrixOp::OpApi op_api) {
    switch(op_api) {
        case HlkMatrixOp::OpApi::hlk_copy_tile_to_dst:       return new HlkMatrixOp_hlk_copy_tile_to_dst(op_api);
        case HlkMatrixOp::OpApi::hlk_tilize_and_copy_to_dst: return new HlkMatrixOp_hlk_tilize_and_copy_to_dst(op_api);
        case HlkMatrixOp::OpApi::hlk_untilize_and_copy_to_dst: return new HlkMatrixOp_hlk_untilize_and_copy_to_dst(op_api);
        case HlkMatrixOp::OpApi::hlk_load_mm_partial_to_dst: return new HlkMatrixOp_hlk_load_mm_partial_to_dst(op_api);
        case HlkMatrixOp::OpApi::hlk_transpose_xy_tile:      return new HlkMatrixOp_hlk_transpose_xy_tile(op_api);
        case HlkMatrixOp::OpApi::hlk_mm_tile:                return new HlkMatrixOp_hlk_mm_tile(op_api);
        case HlkMatrixOp::OpApi::hlk_add_tile:               return new HlkMatrixOp_hlk_add_tile(op_api);
        case HlkMatrixOp::OpApi::hlk_add_tile_bcast:         return new HlkMatrixOp_hlk_add_tile_bcast(op_api);
        case HlkMatrixOp::OpApi::hlk_add_tile_from_dst:      return new HlkMatrixOp_hlk_add_tile_from_dst(op_api);
        case HlkMatrixOp::OpApi::hlk_add_tile_from_dst_bcast: return new HlkMatrixOp_hlk_add_tile_from_dst_bcast(op_api);
        case HlkMatrixOp::OpApi::hlk_multiply_tile:          return new HlkMatrixOp_hlk_multiply_tile(op_api);
        case HlkMatrixOp::OpApi::hlk_multiply_tile_bcast:    return new HlkMatrixOp_hlk_multiply_tile_bcast(op_api);
        case HlkMatrixOp::OpApi::hlk_multiply_tile_from_dst: return new HlkMatrixOp_hlk_multiply_tile_from_dst(op_api);
        case HlkMatrixOp::OpApi::hlk_multiply_tile_from_dst_bcast: return new HlkMatrixOp_hlk_multiply_tile_from_dst_bcast(op_api);
        case HlkMatrixOp::OpApi::hlk_subtract_tile:          return new HlkMatrixOp_hlk_subtract_tile(op_api);
        case HlkMatrixOp::OpApi::hlk_subtract_tile_bcast:    return new HlkMatrixOp_hlk_subtract_tile_bcast(op_api);
        case HlkMatrixOp::OpApi::hlk_subtract_tile_from_dst: return new HlkMatrixOp_hlk_subtract_tile_from_dst(op_api);
        case HlkMatrixOp::OpApi::hlk_subtract_tile_from_dst_bcast: return new HlkMatrixOp_hlk_subtract_tile_from_dst_bcast(op_api);
        case HlkMatrixOp::OpApi::hlk_reduce_tile:            return new HlkMatrixOp_hlk_reduce_tile(op_api);
        case HlkMatrixOp::OpApi::hlk_broadcast_tile:         return new HlkMatrixOp_hlk_broadcast_tile(op_api);
        case HlkMatrixOp::OpApi::hlk_add_tile_to_dst:       return new HlkMatrixOp_hlk_add_tile_to_dst(op_api);
        default:
            assert(false && "Error: unsupported HlkMatrixOp");
    }
}

class HlkSfpuOp : public HlkOp {
    public:
        static constexpr int NUM_HLK_SFPU_OPS = 9;

        enum OpApi : int {
            hlk_sfpu_op_UNINITIALIZED = -1,
            hlk_sfpu_exponential = 0,
            hlk_sfpu_sqrt = 1,
            hlk_sfpu_gelu = 2,
            hlk_sfpu_gelu_derivative = 3,
            hlk_sfpu_reciprocal = 4,
            hlk_sfpu_log = 5,
            hlk_sfpu_tanh = 6,
            hlk_sfpu_dropout = 7,
            hlk_sfpu_sigmoid = 8
        };

        static constexpr OpApi all_op_apis[NUM_HLK_SFPU_OPS] = {
            hlk_sfpu_exponential,
            hlk_sfpu_sqrt,
            hlk_sfpu_gelu,
            hlk_sfpu_gelu_derivative,
            hlk_sfpu_reciprocal,
            hlk_sfpu_log,
            hlk_sfpu_tanh,
            hlk_sfpu_dropout,
            hlk_sfpu_sigmoid
        };

        static const string all_op_api_strs[NUM_HLK_SFPU_OPS];

    protected:
        OpApi op_api;
        string op_str;

        // LLK math
        string math_include_header_str;
        string math_init_func_str;
        string math_func_str;
        vector<int> math_func_hlk_api_args_positions_to_keep;
        vector<int> math_func_hlk_api_option_positions;

        HlkSfpuOp() :
            op_api(hlk_sfpu_op_UNINITIALIZED),
            op_str(""),

            // LLK math
            math_include_header_str(""),
            math_init_func_str(""),
            math_func_str(""),
            math_func_hlk_api_args_positions_to_keep({}),
            math_func_hlk_api_option_positions({}) {}

    public:

        virtual HlkSfpuOp::OpApi get_op_api() {
            return op_api;
        }
        virtual string get_op_str() const override {
            return op_str;
        }

        virtual string get_math_include_header_str() {
            return math_include_header_str;
        }
        virtual string get_math_init_func_str() {
            return math_init_func_str;
        }
        virtual string get_math_func_str() override {
            return math_func_str;
        }
        virtual vector<int> get_math_func_hlk_api_args_positions_to_keep() const override {
            return math_func_hlk_api_args_positions_to_keep;
        }
        virtual vector<int> get_math_func_hlk_api_option_positions() const override {
            return math_func_hlk_api_option_positions;
        }
        virtual void specialize(const vector<int>& int_arg_vals, bool adv_features_en, bool fp32_dest_acc_en) = 0;

        virtual void specialize_math_func_str(const string& dst_mode_str, bool adv_features_en, bool fp32_dest_acc_en) {};

        static HlkSfpuOp *make_hlk_sfpu_op(HlkSfpuOp::OpApi op_api);
        static string get_op_str(HlkSfpuOp::OpApi op_api);
        static void consistency_check();
};

const string HlkSfpuOp::all_op_api_strs[NUM_HLK_SFPU_OPS] = {
    "hlk_sfpu_exponential",
    "hlk_sfpu_sqrt",
    "hlk_sfpu_gelu",
    "hlk_sfpu_gelu_derivative",
    "hlk_sfpu_reciprocal",
    "hlk_sfpu_log",
    "hlk_sfpu_tanh",
    "hlk_sfpu_dropout",
    "hlk_sfpu_sigmoid"
 };

string HlkSfpuOp::get_op_str(HlkSfpuOp::OpApi op_api) {
    return all_op_api_strs[op_api];
}

void HlkSfpuOp::consistency_check() {
    int num_op_apis = 0;
    int num_strs = 0;

    for (auto op_api : all_op_apis) num_op_apis++;
    for (auto strs : all_op_api_strs) num_strs++;

    assert(num_op_apis > 0 && "Error: the number of HLK sfpu op types must be > 0.");
    assert(num_op_apis == num_strs && "Error: the number of HLK sfpu op types and corresponding strings must match.");
}

class HlkSfpuOp_hlk_sfpu_exponential : public HlkSfpuOp {
    public:
        HlkSfpuOp_hlk_sfpu_exponential(HlkSfpuOp::OpApi arg_op_api) : HlkSfpuOp() {
            assert(arg_op_api == HlkSfpuOp::OpApi::hlk_sfpu_exponential);

            op_api = arg_op_api;
            op_str = "hlk_sfpu_exponential";

            math_include_header_str = "llk_math_eltwise_unary_sfpu.h";
            math_init_func_str = "llk_math_eltwise_unary_sfpu_exponential_init";
            math_func_str = "llk_math_eltwise_unary_sfpu_exponential";
            math_func_hlk_api_args_positions_to_keep = {1};
            math_func_hlk_api_option_positions = {};
        }

        virtual void specialize(const vector<int>& int_arg_vals = {}, bool adv_features_en = false, bool fp32_dest_acc_en = false) {

            math_init_func_str += "<APPROX>";
            math_func_str += "<APPROX,";
        }

        virtual void specialize_math_func_str(const string& dst_mode_str, bool adv_features_en, bool fp32_dest_acc_en) { math_func_str += " " + dst_mode_str + ">"; }
};

class HlkSfpuOp_hlk_sfpu_sqrt : public HlkSfpuOp {
    public:
        HlkSfpuOp_hlk_sfpu_sqrt(HlkSfpuOp::OpApi arg_op_api) : HlkSfpuOp() {
            assert(arg_op_api == HlkSfpuOp::OpApi::hlk_sfpu_sqrt);

            op_api = arg_op_api;
            op_str = "hlk_sfpu_sqrt";

            math_include_header_str = "llk_math_eltwise_unary_sfpu.h";
            math_init_func_str = "llk_math_eltwise_unary_sfpu_sqrt_init";
            math_func_str = "llk_math_eltwise_unary_sfpu_sqrt";
            math_func_hlk_api_args_positions_to_keep = {1};
            math_func_hlk_api_option_positions = {};
        }

        virtual void specialize(const vector<int>& int_arg_vals = {}, bool adv_features_en = false, bool fp32_dest_acc_en = false) {

            math_init_func_str += "<APPROX>";
            math_func_str += "<APPROX,";
        }

        virtual void specialize_math_func_str(const string& dst_mode_str, bool adv_features_en, bool fp32_dest_acc_en) { math_func_str += " " + dst_mode_str + ">"; }
};

class HlkSfpuOp_hlk_sfpu_gelu : public HlkSfpuOp {
    public:
        HlkSfpuOp_hlk_sfpu_gelu(HlkSfpuOp::OpApi arg_op_api) : HlkSfpuOp() {
            assert(arg_op_api == HlkSfpuOp::OpApi::hlk_sfpu_gelu);

            op_api = arg_op_api;
            op_str = "hlk_sfpu_gelu";

            math_include_header_str = "llk_math_eltwise_unary_sfpu.h";
            math_init_func_str = "llk_math_eltwise_unary_sfpu_gelu_init";
            math_func_str = "llk_math_eltwise_unary_sfpu_gelu";
            math_func_hlk_api_args_positions_to_keep = {1};
            math_func_hlk_api_option_positions = {};
        }

        virtual void specialize(const vector<int>& int_arg_vals = {}, bool adv_features_en = false, bool fp32_dest_acc_en = false) {

            math_init_func_str += "<APPROX>";
            math_func_str += "<APPROX,";
        }

        virtual void specialize_math_func_str(const string& dst_mode_str, bool adv_features_en, bool fp32_dest_acc_en) { math_func_str += " " + dst_mode_str + ">"; }
};

class HlkSfpuOp_hlk_sfpu_gelu_derivative : public HlkSfpuOp {
    public:
        HlkSfpuOp_hlk_sfpu_gelu_derivative(HlkSfpuOp::OpApi arg_op_api) : HlkSfpuOp() {
            assert(arg_op_api == HlkSfpuOp::OpApi::hlk_sfpu_gelu_derivative);

            op_api = arg_op_api;
            op_str = "hlk_sfpu_gelu_derivative";

            math_include_header_str = "llk_math_eltwise_unary_sfpu.h";
            math_init_func_str = "llk_math_eltwise_unary_sfpu_gelu_derivative_init";
            math_func_str = "llk_math_eltwise_unary_sfpu_gelu_derivative";
            math_func_hlk_api_args_positions_to_keep = {1};
            math_func_hlk_api_option_positions = {};
        }

        virtual void specialize(const vector<int>& int_arg_vals = {}, bool adv_features_en = false, bool fp32_dest_acc_en = false) {

            math_init_func_str += "<APPROX>";
            math_func_str += "<APPROX,";
        }

        virtual void specialize_math_func_str(const string& dst_mode_str, bool adv_features_en, bool fp32_dest_acc_en) { math_func_str += " " + dst_mode_str + ">"; }
};

class HlkSfpuOp_hlk_sfpu_reciprocal : public HlkSfpuOp {
    public:
        HlkSfpuOp_hlk_sfpu_reciprocal(HlkSfpuOp::OpApi arg_op_api) : HlkSfpuOp() {
            assert(arg_op_api == HlkSfpuOp::OpApi::hlk_sfpu_reciprocal);

            op_api = arg_op_api;
            op_str = "hlk_sfpu_reciprocal";

            math_include_header_str = "llk_math_eltwise_unary_sfpu.h";
            math_init_func_str = "llk_math_eltwise_unary_sfpu_reciprocal_init";
            math_func_str = "llk_math_eltwise_unary_sfpu_reciprocal";
            math_func_hlk_api_args_positions_to_keep = {1};
        }

        virtual void specialize(const vector<int>& int_arg_vals = {}, bool adv_features_en = false, bool fp32_dest_acc_en = false) {

            math_init_func_str += "<APPROX>";
            math_func_str += "<APPROX,";
        }

        virtual void specialize_math_func_str(const string& dst_mode_str, bool adv_features_en, bool fp32_dest_acc_en) { math_func_str += " " + dst_mode_str + ">"; }
};

class HlkSfpuOp_hlk_sfpu_log : public HlkSfpuOp {
    public:
        HlkSfpuOp_hlk_sfpu_log(HlkSfpuOp::OpApi arg_op_api) : HlkSfpuOp() {
            assert(arg_op_api == HlkSfpuOp::OpApi::hlk_sfpu_log);

            op_api = arg_op_api;
            op_str = "hlk_sfpu_log";

            math_include_header_str = "llk_math_eltwise_unary_sfpu.h";
            math_init_func_str = "llk_math_eltwise_unary_sfpu_log_init";
            math_func_str = "llk_math_eltwise_unary_sfpu_log";
            math_func_hlk_api_args_positions_to_keep = {1};
            math_func_hlk_api_option_positions = {};
        }

        virtual void specialize(const vector<int>& int_arg_vals = {}, bool adv_features_en = false, bool fp32_dest_acc_en = false) {

            math_init_func_str += "<APPROX>";
            math_func_str += "<APPROX,";
        }

        virtual void specialize_math_func_str(const string& dst_mode_str, bool adv_features_en, bool fp32_dest_acc_en) { math_func_str += " " + dst_mode_str + ">"; }
};

class HlkSfpuOp_hlk_sfpu_tanh : public HlkSfpuOp {
    public:
        HlkSfpuOp_hlk_sfpu_tanh(HlkSfpuOp::OpApi arg_op_api) : HlkSfpuOp() {
            assert(arg_op_api == HlkSfpuOp::OpApi::hlk_sfpu_tanh);

            op_api = arg_op_api;
            op_str = "hlk_sfpu_tanh";

            math_include_header_str = "llk_math_eltwise_unary_sfpu.h";
            math_init_func_str = "llk_math_eltwise_unary_sfpu_tanh_init";
            math_func_str = "llk_math_eltwise_unary_sfpu_tanh";
            math_func_hlk_api_args_positions_to_keep = {1};
            math_func_hlk_api_option_positions = {};
        }

        virtual void specialize(const vector<int>& int_arg_vals = {}, bool adv_features_en = false, bool fp32_dest_acc_en = false) {

            math_init_func_str += "<APPROX>";
            math_func_str += "<APPROX,";
        }

        virtual void specialize_math_func_str(const string& dst_mode_str, bool adv_features_en, bool fp32_dest_acc_en) { math_func_str += " " + dst_mode_str + ">"; }
};

class HlkSfpuOp_hlk_sfpu_dropout : public HlkSfpuOp {
    public:
        HlkSfpuOp_hlk_sfpu_dropout(HlkSfpuOp::OpApi arg_op_api) : HlkSfpuOp() {
            assert(arg_op_api == HlkSfpuOp::OpApi::hlk_sfpu_dropout);

            op_api = arg_op_api;
            op_str = "hlk_sfpu_dropout";

            math_include_header_str = "llk_math_eltwise_unary_sfpu.h";
            math_init_func_str = "llk_math_eltwise_unary_sfpu_dropout_init";
            math_func_str = "llk_math_eltwise_unary_sfpu_dropout";
            math_func_hlk_api_args_positions_to_keep = {1, 2};
            math_func_hlk_api_option_positions = {};
        }

        virtual void specialize(const vector<int>& int_arg_vals = {}, bool adv_features_en = false, bool fp32_dest_acc_en = false) {
            math_func_str += "<APPROX,";
        }

        virtual void specialize_math_func_str(const string& dst_mode_str, bool adv_features_en, bool fp32_dest_acc_en) { math_func_str += " " + dst_mode_str + ">"; }
};

class HlkSfpuOp_hlk_sfpu_sigmoid : public HlkSfpuOp {
    // Not yet supported in llk, but is supported in model
    public:
        HlkSfpuOp_hlk_sfpu_sigmoid(HlkSfpuOp::OpApi arg_op_api) : HlkSfpuOp() {
            assert(arg_op_api == HlkSfpuOp::OpApi::hlk_sfpu_sigmoid);

            op_api = arg_op_api;
            op_str = "hlk_sfpu_sigmoid";

            math_include_header_str = "llk_math_eltwise_unary_sfpu.h";
            math_init_func_str = "llk_math_eltwise_unary_sfpu_sigmoid_init";
            math_func_str = "llk_math_eltwise_unary_sfpu_sigmoid";
            math_func_hlk_api_args_positions_to_keep = {1};
            math_func_hlk_api_option_positions = {};
        }

        virtual void specialize(const vector<int>& int_arg_vals = {}, bool adv_features_en = false, bool fp32_dest_acc_en = false) {

            math_init_func_str += "<APPROX>";
            math_func_str += "<APPROX,";
        }

        virtual void specialize_math_func_str(const string& dst_mode_str, bool adv_features_en, bool fp32_dest_acc_en) { math_func_str += " " + dst_mode_str + ">"; }
};
