import glob
import sys

do_profilize = True
#do_profilize = False

def profilize(lines):
    # converts code like this:
    #   a() // PROF_BEGIN("str")
    #   b() // PROF_BEGIN("str1") PROF_END("str1")
    #   c() // PROF_END("str")
    # to:
    #   PROF_BEGIN("str")
    #   a()
    #   PROF_BEGIN("str1")
    #   b()
    #   PROF_END("str1")
    #   c()
    #   PROF_END("str")
    lines_out = []
    for l in lines:
        begin_pos = l.find("PROF_BEGIN")
        if begin_pos == 0: # beginning of line => no comment before - skip
            begin_pos = -1
        end_pos = l.find("PROF_END")
        if end_pos == 0: # beginning of line => no comment before - skip
            end_pos = -1
        line_out = l
        if begin_pos != -1:
            if end_pos != -1:
                lines_out.append(l[begin_pos:end_pos] + "\n")
            else:
                lines_out.append(l[begin_pos:])
            line_out = l[:begin_pos] + "\n"
        elif end_pos != -1:
            line_out = l[:end_pos] + "\n"
        lines_out.append(line_out)
        if end_pos != -1:
            lines_out.append(l[end_pos:])

    return lines_out

def unprofilize(lines):
    # undoes the effect of profilize
    prev_begin = ""
    out_lines = []
    for il in range(len(lines)):
        l = lines[il]
        if l.find("//") != -1 and prev_begin == "":
            out_lines.append(l)
            continue
        begin_pos = l.find("PROF_BEGIN")
        if begin_pos != -1:
            assert(l.find("PROF_END") == -1)
            prev_begin = l
            continue

        if prev_begin != "":
            assert(l[-5:-1].find("//") != -1) # line must be ending with a comment +/- a few spaces
            l = l[:-1] + prev_begin
            prev_begin = ""
        end_pos = l.find("PROF_END")
        if end_pos != -1:
            prev_line = out_lines[-1]
            begin_added = prev_line.find("PROF_BEGIN") != -1
            assert(begin_added or prev_line[-5:-1].find("//") != -1) # line must be ending with a comment +/- a few spaces
            # strip \n from previous line and append PROF_END
            out_lines[-1] = out_lines[-1][:-1] + l
            continue
        out_lines.append(l)

    return out_lines

if __name__ == "__main__":
    prefix = "./"
    cpp_files = glob.glob(prefix+"**/*.cpp", recursive=True)
    h_files = glob.glob(prefix+"**/*.h", recursive=True)
    all_files = cpp_files+h_files

    for fname in all_files:
        fin = open(fname, "r")
        lines = fin.readlines()
        fin.close()
        if do_profilize == False or (len(sys.argv) > 1 and sys.argv[1] == "-u"):
            lines_out = unprofilize(lines)
        else:
            lines_out = profilize(lines)

        if lines_out != lines:
            fout = open(fname, "wt")
            fout.writelines(lines_out)
            fout.close()
