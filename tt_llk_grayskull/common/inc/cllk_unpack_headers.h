

//
// LLK unpack tile A
// 
inline void llk_unpack_A_hw_config(llk_unpack_A_params_t params);
inline void llk_unpack_A_init();
inline void llk_unpack_A(std::uint32_t tile_index);

//
// LLK unpack tile B
// 
inline void llk_unpack_B_hw_config(llk_unpack_B_params_t params);
inline void llk_unpack_B_init();
inline void llk_unpack_B(std::uint32_t tile_index);

//
// LLK unpack tiles AB
// 
inline void llk_unpack_AB_hw_config(llk_unpack_AB_params_t params);
inline void llk_unpack_AB_init();
inline void llk_unpack_AB(std::uint32_t tile_index_a, std::uint32_t tile_index_b);
