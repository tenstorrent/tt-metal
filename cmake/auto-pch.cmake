# Define global variable for PCH generator file path
set(PCH_GENERATOR_FILE "${CMAKE_BINARY_DIR}/pch_generator.py")
set(COMPILE_COMMANDS_JSON "${CMAKE_BINARY_DIR}/compile_commands.json")

# Clear the PCH generator file when this module is loaded
file(WRITE ${PCH_GENERATOR_FILE} "import sys\nsys.path.append('${CMAKE_SOURCE_DIR}/scripts')\nfrom auto_pch import generate_pch\n\ncompile_commands_json = '${COMPILE_COMMANDS_JSON}'\n\n")
message(STATUS "Cleared PCH generator file: ${PCH_GENERATOR_FILE}")

function(target_auto_pch target_name)
    # Parse named arguments
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "PCH_FILE" "AVOID_INCLUDES")
    
    if(NOT ARG_PCH_FILE)
        message(FATAL_ERROR "target_auto_pch: PCH_FILE argument is required")
    endif()
    
    set(pch_file_path ${ARG_PCH_FILE})
    
    # Get source files from the target
    get_target_property(target_sources ${target_name} SOURCES)
    get_target_property(target_source_dir ${target_name} SOURCE_DIR)
    
    # Set default AVOID_INCLUDES to target's source directory if not provided
    if(NOT ARG_AVOID_INCLUDES)
        set(ARG_AVOID_INCLUDES ${target_source_dir})
    endif()
    
    # Get include directories from the target
    get_target_property(target_includes ${target_name} INCLUDE_DIRECTORIES)
    get_target_property(target_interface_includes ${target_name} INTERFACE_INCLUDE_DIRECTORIES)
    
    # Initialize lists
    set(source_files_list "")
    set(avoid_includes_list "")
    
    # Process source files - convert relative paths to absolute
    if(target_sources)
        foreach(source ${target_sources})
            if(IS_ABSOLUTE ${source})
                list(APPEND source_files_list ${source})
            else()
                list(APPEND source_files_list ${target_source_dir}/${source})
            endif()
        endforeach()
    endif()
    
    # Process avoid includes - convert to absolute paths
    foreach(avoid_path ${ARG_AVOID_INCLUDES})
        if(IS_ABSOLUTE ${avoid_path})
            list(APPEND avoid_includes_list ${avoid_path})
        else()
            list(APPEND avoid_includes_list ${target_source_dir}/${avoid_path})
        endif()
    endforeach()
    
    # Remove duplicates
    if(avoid_includes_list)
        list(REMOVE_DUPLICATES avoid_includes_list)
    endif()
    
    # Resolve pch file path to absolute
    if(IS_ABSOLUTE ${pch_file_path})
        set(absolute_pch_path ${pch_file_path})
    else()
        set(absolute_pch_path ${target_source_dir}/${pch_file_path})
    endif()
    
    # Format lists for Python
    set(python_source_files "source_files = [\n")
    if(source_files_list)
        foreach(source ${source_files_list})
            string(APPEND python_source_files "    \"${source}\",\n")
        endforeach()
    endif()
    string(APPEND python_source_files "]\n\n")
    
    set(python_avoid_includes "avoid_includes = [\n")
    if(avoid_includes_list)
        foreach(avoid_include ${avoid_includes_list})
            string(APPEND python_avoid_includes "    \"${avoid_include}\",\n")
        endforeach()
    endif()
    string(APPEND python_avoid_includes "]\n\n")

    set(python_pch_file "pch_file = \"${absolute_pch_path}\"\n\n")
    set(python_write "print(\"${target_name}: ${absolute_pch_path}\")\n")
    set(python_call "generate_pch(compile_commands_json, source_files, avoid_includes, pch_file)\n\n")

    # Create the Python content section
    set(python_section "# Target: ${target_name}\n${python_source_files}${python_avoid_includes}${python_pch_file}${python_write}${python_call}")

    # Append to the Python file
    file(APPEND ${PCH_GENERATOR_FILE} "${python_section}")

    message(STATUS "Added PCH information for target '${target_name}' to ${PCH_GENERATOR_FILE}")
endfunction()
