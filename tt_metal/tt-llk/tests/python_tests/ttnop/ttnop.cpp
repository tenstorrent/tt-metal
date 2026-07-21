#include <elf.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "include/util.h"

// A human label for an opcode, used by --list and --every class matching.
const char* insn_class(std::uint32_t insn)
{
    switch (opcode(insn))
    {
        case OP_LOAD:
            return "load";
        case OP_OPIMM:
            return (insn == NOP) ? "nop" : "opimm";
        case OP_AUIPC:
            return "auipc";
        case OP_STORE:
            return "store";
        case OP_AMO:
            return "amo";
        case OP_OP:
            return "op";
        case OP_LUI:
            return "lui";
        case OP_BRANCH:
            return "branch";
        case OP_JALR:
            return "jalr";
        case OP_JAL:
            return (rd(insn) == 0) ? "jump" : "call";
        case OP_SYSTEM:
            return "system";
        case OP_MISCMEM:
            return "fence";
        // The four RISC-V "custom" opcode slots carry the Tensix coprocessor ops.
        case 0x0b:
        case 0x2b:
        case 0x5b:
        case 0x7b:
            return "custom";
        default:
            return "other";
    }
}

// ----------------------------------------------------------------- ELF model
struct Section
{
    Elf32_Shdr h {};
    std::string name;
    int index = 0;
};

struct Sym
{
    std::uint32_t value = 0;
    std::string name;
    std::uint8_t type = 0;
};

struct Elf
{
    std::string path;              // input path (for permission preservation)
    std::vector<std::uint8_t> buf; // whole file image
    Elf32_Ehdr eh {};
    std::vector<Section> secs;
    std::vector<Elf32_Phdr> phdrs;
    std::map<std::string, Sym> syms; // by name (first definition wins)
    int text_idx  = -1;
    int exec_phdr = -1; // the PF_X PT_LOAD containing .text

    const Section& sec(int i) const
    {
        return secs.at(i);
    }
};

template <class T>
T rd_at(const std::vector<std::uint8_t>& b, size_t off)
{
    T v {};
    if (off + sizeof(T) > b.size())
    {
        die("truncated ELF (read past end)");
    }
    std::memcpy(&v, b.data() + off, sizeof(T));
    return v;
}

void load_elf(const std::string& path, Elf& e)
{
    e.path = path;
    std::ifstream f(path, std::ios::binary);
    if (!f)
    {
        die("cannot open " + path);
    }
    e.buf.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (e.buf.size() < sizeof(Elf32_Ehdr))
    {
        die("file too small to be an ELF");
    }

    const std::uint8_t* id = e.buf.data();
    if (std::memcmp(id, ELFMAG, SELFMAG) != 0)
    {
        die("not an ELF file");
    }
    if (id[EI_CLASS] != ELFCLASS32)
    {
        die("not ELF32 (only RV32 kernels supported)");
    }
    if (id[EI_DATA] != ELFDATA2LSB)
    {
        die("not little-endian");
    }
    if (!host_is_le())
    {
        die("host is big-endian; unsupported");
    }

    e.eh = rd_at<Elf32_Ehdr>(e.buf, 0);
    if (e.eh.e_machine != EM_RISCV)
    {
        die("e_machine is not RISC-V");
    }
    if (e.eh.e_type != ET_EXEC)
    {
        die("not a linked executable (need ET_EXEC)");
    }

    for (std::uint32_t i = 0; i < e.eh.e_phnum; ++i)
    {
        e.phdrs.push_back(rd_at<Elf32_Phdr>(e.buf, e.eh.e_phoff + i * e.eh.e_phentsize));
    }

    for (std::uint32_t i = 0; i < e.eh.e_shnum; ++i)
    {
        Section s;
        s.index = static_cast<int>(i);
        s.h     = rd_at<Elf32_Shdr>(e.buf, e.eh.e_shoff + i * e.eh.e_shentsize);
        e.secs.push_back(s);
    }
    if (e.eh.e_shstrndx >= e.secs.size())
    {
        die("bad e_shstrndx");
    }
    const Elf32_Shdr& shstr = e.secs[e.eh.e_shstrndx].h;
    for (auto& s : e.secs)
    {
        size_t off = shstr.sh_offset + s.h.sh_name;
        if (off >= e.buf.size())
        {
            die("section name out of range");
        }
        s.name = reinterpret_cast<const char*>(e.buf.data() + off);
    }

    // Locate .text and the executable PT_LOAD that contains it.
    for (auto& s : e.secs)
    {
        if (s.name == ".text" && (s.h.sh_flags & SHF_EXECINSTR))
        {
            e.text_idx = s.index;
        }
    }
    if (e.text_idx < 0)
    {
        for (auto& s : e.secs)
        {
            if ((s.h.sh_flags & SHF_EXECINSTR) && s.h.sh_type == SHT_PROGBITS)
            {
                e.text_idx = s.index;
                break;
            }
        }
    }
    if (e.text_idx < 0)
    {
        die("no executable .text section found");
    }
    const Section& t = e.sec(e.text_idx);
    for (size_t i = 0; i < e.phdrs.size(); ++i)
    {
        const Elf32_Phdr& p = e.phdrs[i];
        if (p.p_type == PT_LOAD && (p.p_flags & PF_X) && t.h.sh_addr >= p.p_vaddr && t.h.sh_addr + t.h.sh_size <= p.p_vaddr + p.p_memsz)
        {
            e.exec_phdr = static_cast<int>(i);
        }
    }
    if (e.exec_phdr < 0)
    {
        die("no executable PT_LOAD segment contains .text");
    }

    // We re-serialize the file (the grown .text shifts later sections), which moves
    // file offsets. That is only safe if the executable segment does not itself hold
    // the ELF header / program-header table. Linked kernels keep .text in its own
    // segment past the headers; host `gcc -static` does not, so refuse that.
    {
        const Elf32_Phdr& xp = e.phdrs[e.exec_phdr];
        std::uint32_t ph_end = e.eh.e_phoff + std::uint32_t(e.eh.e_phnum) * e.eh.e_phentsize;
        bool covers_hdr      = xp.p_offset == 0;
        bool covers_phdrs    = xp.p_offset <= e.eh.e_phoff && e.eh.e_phoff < xp.p_offset + xp.p_filesz;
        if (covers_hdr || (covers_phdrs && xp.p_offset < ph_end))
        {
            die("the executable segment contains the ELF/program headers; cannot grow "
                "it safely (normal for host `gcc -static`, not for linked kernels)");
        }
    }

    // Symbols (.symtab preferred; .dynsym fallback), for symbol/symbol+off targeting.
    auto load_symtab = [&](std::uint32_t shtype)
    {
        for (auto& s : e.secs)
        {
            if (s.h.sh_type != shtype)
            {
                continue;
            }
            const Section& str = e.sec(s.h.sh_link);
            std::uint32_t n    = s.h.sh_entsize ? s.h.sh_size / s.h.sh_entsize : 0;
            for (std::uint32_t i = 0; i < n; ++i)
            {
                Elf32_Sym sym = rd_at<Elf32_Sym>(e.buf, s.h.sh_offset + i * sizeof(Elf32_Sym));
                if (!sym.st_name)
                {
                    continue;
                }
                std::string name = reinterpret_cast<const char*>(e.buf.data() + str.h.sh_offset + sym.st_name);
                if (name.empty() || e.syms.count(name))
                {
                    continue;
                }
                Sym o;
                o.value      = sym.st_value;
                o.name       = name;
                o.type       = ELF32_ST_TYPE(sym.st_info);
                e.syms[name] = o;
            }
        }
    };
    load_symtab(SHT_SYMTAB);
    if (e.syms.empty())
    {
        load_symtab(SHT_DYNSYM);
    }
}

std::uint32_t text_word(const Elf& e, std::uint32_t vaddr)
{
    const Section& t = e.sec(e.text_idx);
    return rd_at<std::uint32_t>(e.buf, t.h.sh_offset + (vaddr - t.h.sh_addr));
}

// ------------------------------------------------------------------- patches
struct Site
{
    std::uint32_t vaddr = 0; // address of the instruction we detour
    std::uint32_t n     = 0; // delay NOP count
    std::string why;         // provenance, for the report
};

struct Block
{
    std::uint32_t site       = 0;
    std::uint32_t orig       = 0;
    std::uint32_t n          = 0;
    std::uint32_t cave_vaddr = 0;     // where this block lives
    std::vector<std::uint32_t> words; // body of the cave block
    std::uint32_t patch = 0;          // the `jal x0, cave` that replaces orig
};

// Words after the delay: branch needs 3 (long-form), everything else needs 2.
std::uint32_t body_words(std::uint32_t orig)
{
    return opcode(orig) == OP_BRANCH ? 3 : 2;
}

void build_block(Block& b)
{
    const std::uint32_t A = b.site;
    const std::uint32_t I = b.orig;
    const std::uint32_t C = b.cave_vaddr;
    auto& w               = b.words;
    for (std::uint32_t k = 0; k < b.n; ++k)
    {
        w.push_back(NOP);
    }
    std::uint32_t pc = C + b.n * 4; // vaddr of the first post-delay word

    switch (opcode(I))
    {
        case OP_JAL:
        {
            // Preserve the link register; recompute the target for the cave PC.
            std::uint32_t target = A + static_cast<std::uint32_t>(jal_imm(I));
            if (!jal_in_range(pc, target))
            {
                die("relocated jal out of +/-1MiB range");
            }
            w.push_back(make_jal(rd(I), pc, target));
            // Return trampoline (taken when rd!=x0, i.e. a call returns here).
            if (!jal_in_range(pc + 4, A + 4))
            {
                die("cave return jal out of range");
            }
            w.push_back(make_jal(0, pc + 4, A + 4));
            break;
        }
        case OP_BRANCH:
        {
            // Distance-independent long form:
            //   B!cc rs1,rs2, +8   ; skip the taken jal if the branch would NOT take
            //   jal  x0, target    ; taken edge
            //   jal  x0, A+4        ; fall-through edge
            std::uint32_t target = A + static_cast<std::uint32_t>(branch_imm(I));
            std::uint32_t inv    = OP_BRANCH | ((funct3(I) ^ 1) << 12) | (rs1(I) << 15) | (rs2(I) << 20) | enc_branch_imm(8);
            w.push_back(inv);
            if (!jal_in_range(pc + 4, target))
            {
                die("branch taken-edge jal out of range");
            }
            w.push_back(make_jal(0, pc + 4, target));
            if (!jal_in_range(pc + 8, A + 4))
            {
                die("branch fall-edge jal out of range");
            }
            w.push_back(make_jal(0, pc + 8, A + 4));
            break;
        }
        default:
        {
            // Position-independent: copy verbatim, then return. (loads/stores/ALU/
            // atomics/csr/fence/jalr and Tensix custom ops. A jalr writing ra lands
            // its return on the trampoline.)
            w.push_back(I);
            if (!jal_in_range(pc + 4, A + 4))
            {
                die("cave return jal out of range");
            }
            w.push_back(make_jal(0, pc + 4, A + 4));
            break;
        }
    }
    if (!jal_in_range(A, C))
    {
        die("site-to-cave jal out of +/-1MiB range (cave too far)");
    }
    b.patch = make_jal(0, A, C); // jal x0, cave
}

// ---------------------------------------------------------------- serialize
template <class T>
void put(std::vector<std::uint8_t>& v, size_t off, const T& x)
{
    std::memcpy(v.data() + off, &x, sizeof(T));
}

// Re-serialize the file with the cave folded onto the end of .text. .text grows, so
// every later section's file offset is recomputed; virtual addresses are unchanged.
void write_out(const Elf& e, std::vector<Block>& blocks, std::uint32_t cave_vaddr, const std::vector<std::uint8_t>& cave_bytes, const std::string& out_path)
{
    const Section& text    = e.sec(e.text_idx);
    const Elf32_Phdr xseg0 = e.phdrs[e.exec_phdr];
    const int n0           = static_cast<int>(e.secs.size());

    // Patched .text with the cave appended. cave_vaddr is the 4-aligned end of .text,
    // so any gap before it is just padding; the cave sits above all existing code, so
    // no address moves.
    std::vector<std::uint8_t> text_bytes(e.buf.begin() + text.h.sh_offset, e.buf.begin() + text.h.sh_offset + text.h.sh_size);
    for (const Block& b : blocks)
    {
        put(text_bytes, b.site - text.h.sh_addr, b.patch);
    }
    text_bytes.resize(cave_vaddr - text.h.sh_addr, 0);
    text_bytes.insert(text_bytes.end(), cave_bytes.begin(), cave_bytes.end());

    std::vector<Elf32_Shdr> sh;
    for (const Section& s : e.secs)
    {
        sh.push_back(s.h);
    }
    sh[e.text_idx].sh_size = std::uint32_t(text_bytes.size());

    const std::uint32_t exec_filesz = (cave_vaddr - xseg0.p_vaddr) + std::uint32_t(cave_bytes.size());

    // ---- Assign new file offsets.
    std::vector<std::int64_t> noff(sh.size(), -1);
    std::vector<std::uint32_t> seg_off(e.phdrs.size(), 0);
    std::uint32_t cur = sizeof(Elf32_Ehdr) + std::uint32_t(e.eh.e_phnum) * e.eh.e_phentsize;

    auto seg_memsz = [&](int p) { return p == e.exec_phdr ? std::max(e.phdrs[p].p_memsz, exec_filesz) : e.phdrs[p].p_memsz; };

    std::vector<int> loads;
    for (int i = 0; i < int(e.phdrs.size()); ++i)
    {
        if (e.phdrs[i].p_type == PT_LOAD)
        {
            loads.push_back(i);
        }
    }
    std::sort(loads.begin(), loads.end(), [&](int a, int b) { return e.phdrs[a].p_vaddr < e.phdrs[b].p_vaddr; });

    for (int p : loads)
    {
        std::uint32_t pal = e.phdrs[p].p_align ? e.phdrs[p].p_align : 1;
        while (cur % pal != e.phdrs[p].p_vaddr % pal)
        {
            ++cur; // file/vaddr congruence
        }
        seg_off[p]         = cur;
        std::uint32_t base = cur, msz = seg_memsz(p), endf = base;
        for (int i = 0; i < int(sh.size()); ++i)
        {
            if (!(sh[i].sh_flags & SHF_ALLOC))
            {
                continue;
            }
            if (sh[i].sh_addr < e.phdrs[p].p_vaddr || sh[i].sh_addr >= e.phdrs[p].p_vaddr + msz)
            {
                continue;
            }
            noff[i] = base + (sh[i].sh_addr - e.phdrs[p].p_vaddr);
            if (sh[i].sh_type != SHT_NOBITS)
            {
                endf = std::max<std::uint32_t>(endf, noff[i] + sh[i].sh_size);
            }
        }
        cur = std::max(cur, endf);
    }
    // Everything not in a PT_LOAD: .symtab/.strtab/.shstrtab/.debug_*/.comment ...
    for (int i = 0; i < int(sh.size()); ++i)
    {
        if (noff[i] >= 0)
        {
            continue;
        }
        if (sh[i].sh_type == SHT_NULL)
        {
            noff[i] = 0;
            continue;
        }
        if (sh[i].sh_type == SHT_NOBITS)
        {
            noff[i] = cur;
            continue;
        }
        std::uint32_t al = sh[i].sh_addralign ? sh[i].sh_addralign : 1;
        while (cur % al)
        {
            ++cur;
        }
        noff[i] = cur;
        cur += sh[i].sh_size;
    }
    while (cur % 4)
    {
        ++cur;
    }
    std::uint32_t shoff = cur;
    cur += std::uint32_t(sh.size()) * sizeof(Elf32_Shdr);

    // ---- Emit.
    std::vector<std::uint8_t> out(cur, 0);

    Elf32_Ehdr neh = e.eh;
    neh.e_shoff    = shoff;
    neh.e_shnum    = std::uint16_t(sh.size());
    put(out, 0, neh);

    for (int i = 0; i < int(e.phdrs.size()); ++i)
    {
        Elf32_Phdr ph = e.phdrs[i];
        if (ph.p_type == PT_LOAD)
        {
            ph.p_offset = seg_off[i];
            if (i == e.exec_phdr)
            {
                ph.p_filesz = exec_filesz;
                ph.p_memsz  = std::max(ph.p_memsz, exec_filesz);
            }
        }
        else if (ph.p_filesz > 0)
        {
            for (int s = 0; s < n0; ++s)
            {
                if (e.secs[s].h.sh_offset == ph.p_offset)
                {
                    ph.p_offset = std::uint32_t(noff[s]);
                    break;
                }
            }
        }
        put(out, e.eh.e_phoff + i * e.eh.e_phentsize, ph);
    }

    for (int i = 0; i < int(sh.size()); ++i)
    {
        if (noff[i] < 0 || sh[i].sh_type == SHT_NOBITS || sh[i].sh_type == SHT_NULL)
        {
            if (noff[i] >= 0)
            {
                sh[i].sh_offset = std::uint32_t(noff[i]);
            }
            continue;
        }
        const std::uint8_t* src;
        size_t len;
        if (i == e.text_idx)
        {
            src = text_bytes.data();
            len = text_bytes.size();
        }
        else
        {
            src = e.buf.data() + e.secs[i].h.sh_offset;
            len = e.secs[i].h.sh_size;
        }
        if (noff[i] + std::int64_t(len) > std::int64_t(out.size()))
        {
            die("internal: layout overflow");
        }
        std::memcpy(out.data() + noff[i], src, len);
        sh[i].sh_offset = std::uint32_t(noff[i]);
    }

    for (int i = 0; i < int(sh.size()); ++i)
    {
        put(out, shoff + i * sizeof(Elf32_Shdr), sh[i]);
    }

    std::ofstream of(out_path, std::ios::binary | std::ios::trunc);
    if (!of)
    {
        die("cannot write " + out_path);
    }
    of.write(reinterpret_cast<const char*>(out.data()), out.size());
    if (!of)
    {
        die("write failed");
    }
    of.close();
    // Preserve the input's permission bits (notably the execute bit).
    struct stat st;
    if (::stat(e.path.c_str(), &st) == 0)
    {
        ::chmod(out_path.c_str(), st.st_mode & 07777);
    }
}

// ------------------------------------------------------------------ helpers
// Resolve "0x6510", "main", "main+0x10", "main+4" to a .text vaddr.
bool resolve_loc(const Elf& e, const std::string& s, std::uint32_t& vaddr, std::string& err)
{
    std::string base  = s;
    std::uint32_t add = 0;
    size_t plus       = s.find('+');
    if (plus != std::string::npos)
    {
        base = s.substr(0, plus);
        if (!parse_u32(s.substr(plus + 1), add))
        {
            err = "bad offset in '" + s + "'";
            return false;
        }
    }
    std::uint32_t v;
    if (parse_u32(base, v))
    {
        vaddr = v + add;
        return true;
    }
    auto it = e.syms.find(base);
    if (it == e.syms.end())
    {
        err = "unknown symbol '" + base + "'";
        return false;
    }
    vaddr = it->second.value + add;
    return true;
}

const Section* sec_containing(const Elf& e, std::uint32_t vaddr)
{
    for (const Section& s : e.secs)
    {
        if ((s.h.sh_flags & SHF_ALLOC) && vaddr >= s.h.sh_addr && vaddr < s.h.sh_addr + s.h.sh_size)
        {
            return &s;
        }
    }
    return nullptr;
}

void validate_site(const Elf& e, Site& site)
{
    const Section& t = e.sec(e.text_idx);
    char b[160];
    if (site.vaddr < t.h.sh_addr || site.vaddr + 4 > t.h.sh_addr + t.h.sh_size)
    {
        std::snprintf(b, sizeof b, "site 0x%x is outside .text [0x%x,0x%x)", site.vaddr, t.h.sh_addr, t.h.sh_addr + t.h.sh_size);
        die(b);
    }
    if (site.vaddr % 4)
    {
        std::snprintf(b, sizeof b, "site 0x%x is not 4-byte aligned (RVC not supported)", site.vaddr);
        die(b);
    }
    if (opcode(text_word(e, site.vaddr)) == OP_AUIPC)
    {
        std::snprintf(
            b,
            sizeof b,
            "site 0x%x is an auipc; it is PC-relative and cannot "
            "be detoured - pick an adjacent site",
            site.vaddr);
        die(b);
    }
}

// batch mode (OpenMP)
// The `batch` command parses the base kernel ELF ONCE and then,
// in parallel (OpenMP), emits one perturbed ELF *set* per NOP count.
// Only `--thread (either math.elf, unpack.elf, pack.elf)` is patched;
// the other elfs are hardlinked from --base-dir (symlink).
// pytest-xdist (`-n 8`) runs the sets across 8 Tensix cores.

// Make every directory component of `path` (which may end in '/').
static void mkdirs(const std::string& path)
{
    std::string acc;
    for (size_t i = 0; i < path.size(); ++i)
    {
        acc += path[i];
        if (path[i] == '/' || i + 1 == path.size())
        {
            if (acc == "/" || acc.empty())
            {
                continue;
            }
            std::string d = (path[i] == '/') ? acc.substr(0, acc.size() - 1) : acc;
            if (::mkdir(d.c_str(), 0777) != 0 && errno != EEXIST)
            {
                die("mkdir failed: " + d);
            }
        }
    }
}

// Point dst at the unchanged baseline ELF using a hardlink.
// If we are patching math.elf, unpack and pack elf will always be the same across
// the 100*blocks iterations. Copying the same file is not efficient here and creating a
// hardlink is much more efficient because the file is the EXACT same across all iterations
static void link_sibling_elf(const std::string& src, const std::string& dst)
{
    // Remove any stale destination entry first.
    if (::unlink(dst.c_str()) != 0 && errno != ENOENT)
    {
        die("batch: cannot remove stale sibling: " + dst);
    }

    if (::link(src.c_str(), dst.c_str()) != 0)
    {
        die("batch: cannot hardlink sibling " + src + " -> " + dst);
    }
}

// Checks if the virtual address chosen is safe to add NOPs to
static bool is_detourable(const Elf& e, std::uint32_t v)
{
    const Section& t = e.sec(e.text_idx);

    // check if the virtual address is inside .text and room for 4 byte instruction to be inserted
    if (v < t.h.sh_addr || v + 4 > t.h.sh_addr + t.h.sh_size)
    {
        return false;
    }

    // address must also be 4 byte aligned
    if (v % 4)
    {
        return false;
    }

    std::uint32_t oc = opcode(text_word(e, v));

    // if the instruction is unsafe return false
    return oc != OP_AUIPC && oc != OP_BRANCH && oc != OP_JAL;
}

// Batch injection site: entry of run_kernel (past crt0 setup).
// Mangled names still contain "run_kernel"; skip non-FUNC symbols with that substring.
std::uint32_t pick_auto_site(const Elf& e)
{
    // 1) Prefer run_kernel (mangled names still contain "run_kernel").
    for (const auto& pair : e.syms)
    {
        const Sym& s = pair.second;
        // run_kernel must be a function, there can be variables names with it too
        if (s.type != STT_FUNC)
        {
            continue;
        }
        if (s.name.find("run_kernel") == std::string::npos)
        {
            continue;
        }
        if (!is_detourable(e, s.value))
        {
            continue;
        }
        std::fprintf(stderr, "ttnop: site -> run_kernel entry 0x%08x (<%s>)\n", s.value, s.name.c_str());
        return s.value;
    }
    die("no detourable run_kernel function symbol found");
}

// Patch a single already-loaded base ELF with N NOPs at one site and write it out.
// Shares the read-only `base` across OpenMP workers (no re-parse per variant).
void patch_one(const Elf& base, std::uint32_t site_vaddr, std::uint32_t n, const std::string& out_path)
{
    const Section& t = base.sec(base.text_idx);

    // t.h.sh_addr -> where .text starts
    // t.h.sh_size -> size of .text
    // finds the 4 byte aligned address right after .text
    std::uint32_t cave_vaddr = ((t.h.sh_addr + t.h.sh_size) + 3u) & ~3u;

    Block b;
    b.site       = site_vaddr;                  // address where jump will be
    b.orig       = text_word(base, site_vaddr); // current instruction that jump will replace
    b.n          = n;
    b.cave_vaddr = cave_vaddr;

    build_block(b);

    // Take the list of 32-bit words into bytes for write_out
    const std::uint8_t* p = reinterpret_cast<const std::uint8_t*>(b.words.data());
    std::vector<std::uint8_t> cave_bytes(p, p + b.words.size() * 4);

    std::vector<Block> blocks {b};
    write_out(base, blocks, cave_vaddr, cave_bytes, out_path);
}

std::vector<std::string> split_csv(const std::string& s)
{
    std::vector<std::string> out;
    size_t i = 0;
    while (i < s.size())
    {
        size_t j        = s.find(',', i);
        std::string tok = s.substr(i, j == std::string::npos ? std::string::npos : j - i);
        if (!tok.empty())
        {
            out.push_back(tok);
        }
        if (j == std::string::npos)
        {
            break;
        }
        i = j + 1;
    }
    return out;
}

struct BatchArgs
{
    std::string base_dir, out_root, thread = "math";
    std::vector<std::string> components = {"unpack", "math", "pack"};
    std::vector<std::uint32_t> counts;
    int jobs = 0;
};

int do_batch(BatchArgs a)
{
    if (a.base_dir.empty())
    {
        die("batch: --base-dir <elf dir> is required");
    }
    if (a.out_root.empty())
    {
        die("batch: --out-root <dir> is required");
    }
    if (a.counts.empty())
    {
        die("batch: --counts <csv> is required");
    }

    if (std::find(a.components.begin(), a.components.end(), a.thread) == a.components.end())
    {
        die("batch: --thread '" + a.thread + "' not in components");
    }

    // Parse the base thread ELF once; OpenMP workers share it read-only.
    Elf base;
    load_elf(a.base_dir + "/" + a.thread + ".elf", base);

    // Resolve + validate the injection site ONCE (before the parallel region).
    std::uint32_t site_vaddr = pick_auto_site(base);
    Site site_chk;
    site_chk.vaddr = site_vaddr;
    validate_site(base, site_chk);

    // Sibling ELFs are unchanged; workers hardlink them from --base-dir.
    for (const std::string& c : a.components)
    {
        if (c == a.thread)
        {
            continue;
        }
        std::string src = a.base_dir + "/" + c + ".elf";
        struct stat st;
        if (::stat(src.c_str(), &st) != 0)
        {
            die("batch: missing sibling ELF: " + src);
        }
    }

    const int N = static_cast<int>(a.counts.size());
#ifdef _OPENMP
    // Cap OpenMP workers to omp_get_max_threads()
    // so batch threads don't oversubscribe when many xdist workers run at once.
    // If jobs are higher than max_threads, then the OS time-slices them and there are more context switches
    const int max_threads = std::max(1, omp_get_max_threads());
    int jobs              = std::max(1, std::min(N, max_threads));
    if (a.jobs > 0)
    {
        jobs = std::max(1, std::min(jobs, a.jobs));
    }
#else
    // not using openMP set jobs to 1 and ignore -j
    const int jobs = 1;
    (void)a.jobs;
#endif
    std::fprintf(
        stderr,
        "ttnop batch: base_dir=%s thread=%s site=0x%08x (%s) payload=RISCV_NOP variants=%d jobs=%d (max_threads=%d)\n",
        a.base_dir.c_str(),
        a.thread.c_str(),
        site_vaddr,
        insn_class(text_word(base, site_vaddr)),
        N,
        jobs,
#ifdef _OPENMP
        max_threads
#else
        1
#endif
    );

    mkdirs(a.out_root + "/");

#ifdef _OPENMP
    omp_set_num_threads(jobs);
#endif

#pragma omp parallel for schedule(static, 1) // round robin jobs
    for (int i = 0; i < N; ++i)
    {
        std::uint32_t n = a.counts[i];
        std::string dir = a.out_root + "/n" + std::to_string(n);
        mkdirs(dir + "/");
        for (const std::string& c : a.components)
        {
            std::string dst = dir + "/" + c + ".elf";
            if (c == a.thread)
            {
                patch_one(base, site_vaddr, n, dst);
            }
            else
            {
                link_sibling_elf(a.base_dir + "/" + c + ".elf", dst);
            }
        }
    }

    return 0;
}

int run_batch(int argc, char** argv)
{
    BatchArgs a;
    auto take = [&](int& i) -> std::string
    {
        if (i + 1 >= argc)
        {
            die(std::string("missing argument to ") + argv[i]);
        }
        return argv[++i];
    };
    for (int i = 2; i < argc; ++i)
    {
        std::string s = argv[i];
        if (s == "--base-dir")
        {
            a.base_dir = take(i);
        }
        else if (s == "--out-root")
        {
            a.out_root = take(i);
        }
        else if (s == "--thread")
        {
            a.thread = take(i);
        }
        else if (s == "--counts")
        {
            for (const std::string& tok : split_csv(take(i)))
            {
                std::uint32_t v;
                if (!parse_u32(tok, v))
                {
                    die("batch: bad --counts value '" + tok + "'");
                }
                if (std::find(a.counts.begin(), a.counts.end(), v) != a.counts.end())
                {
                    die("batch: duplicate --counts value '" + tok + "'");
                }
                a.counts.push_back(v);
            }
        }
        else if (s == "-j")
        {
            std::uint32_t v;
            if (!parse_u32(take(i), v) || v == 0)
            {
                die("batch: bad -j value (want positive integer)");
            }
            a.jobs = static_cast<int>(v);
        }
        else
        {
            die("batch: unknown option: " + s);
        }
    }
    return do_batch(a);
}

// --------------------------------------------------------------------- main
void usage()
{
    std::printf(
        "ttnop\ninject NOP delays into a linked RV32 kernel ELF\n\n"
        "usage: ttnop <in.elf> -o <out.elf> [SITE ...] [options]\n"
        "       ttnop batch --counts <csv> [options]\n\n"
        "BATCH MODE (OpenMP: parse base once, emit one perturbed ELF set per count):\n"
        "  --counts CSV       one perturbed set per value (the NOP count; no duplicates)\n"
        "  --thread T         component that receives the NOPs (default math)\n"
        "  -j N               cap OpenMP workers (default: OMP_NUM_THREADS / OpenMP max)\n"
        "  (ELF set is always unpack/math/pack; siblings hardlinked from base dir)\n"
        "  (injection site is always run_kernel entry)\n\n"
        "SINGLE-ELF MODE:\n"
        "A SITE is  LOC=N  meaning: inject a delay of N before the instruction at LOC.\n"
        "  LOC is  0xADDR | symbol | symbol+0xOFF | 0xADDR+0xOFF\n"
        "  examples:  0x6510=5   main=8   main+0x1c=3\n\n"
        "options:\n"
        "  -o FILE            output ELF (required unless --list)\n"
        "  -f FILE            read SITE lines from FILE ('#' comments, blank lines ok)\n"
        "  --every CLASS=N    inject a delay of N before every instruction of CLASS\n"
        "                     CLASS: load store op opimm lui branch jal call jalr\n"
        "                            system fence amo custom all\n"
        "  --list             classify .text (with no sites: print the disassembly + tally)\n"
        "  --verify           re-open the output and re-check the detours\n"
        "  -h, --help         this message\n");
}

struct EveryReq
{
    std::string cls;
    std::uint32_t n;
};

int run(int argc, char** argv)
{
    if (argc < 2)
    {
        usage();
        return 1;
    }

    // OpenMP batch sub-command: compile-once / patch-many across NOP counts.
    if (std::string(argv[1]) == "batch")
    {
        return run_batch(argc, argv);
    }

    std::string in_path, out_path, sites_file;
    std::vector<Site> explicit_sites;
    std::vector<EveryReq> every;
    bool do_list = false, do_verify = false;

    auto take = [&](int& i) -> std::string
    {
        if (i + 1 >= argc)
        {
            die(std::string("missing argument to ") + argv[i]);
        }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "-h" || a == "--help")
        {
            usage();
            return 0;
        }
        else if (a == "-o")
        {
            out_path = take(i);
        }
        else if (a == "-f")
        {
            sites_file = take(i);
        }
        else if (a == "--list")
        {
            do_list = true;
        }
        else if (a == "--verify")
        {
            do_verify = true;
        }
        else if (a == "--every")
        {
            std::string v = take(i);
            size_t eq     = v.find('=');
            std::uint32_t n;
            if (eq == std::string::npos || !parse_u32(v.substr(eq + 1), n))
            {
                die("bad --every (want CLASS=N): " + v);
            }
            every.push_back({v.substr(0, eq), n});
        }
        else if (!a.empty() && a[0] == '-')
        {
            die("unknown option: " + a);
        }
        else if (in_path.empty())
        {
            in_path = a;
        }
        else
        {
            // SITE: LOC=N (LOC resolved after the ELF is loaded).
            size_t eq = a.find('=');
            std::uint32_t n;
            if (eq == std::string::npos || !parse_u32(a.substr(eq + 1), n))
            {
                die("bad SITE (want LOC=N): " + a);
            }
            Site s;
            s.n   = n;
            s.why = a.substr(0, eq);
            explicit_sites.push_back(s);
        }
    }
    if (in_path.empty())
    {
        die("no input ELF given");
    }

    Elf e;
    load_elf(in_path, e);
    const Section& t = e.sec(e.text_idx);

    // ---- --list with no sites: classify .text and exit.
    if (do_list && explicit_sites.empty() && every.empty() && sites_file.empty())
    {
        std::printf(".text  vaddr=0x%x  size=0x%x  (%u instructions)\n", t.h.sh_addr, t.h.sh_size, t.h.sh_size / 4);
        std::map<std::string, int> tally;
        for (std::uint32_t v = t.h.sh_addr; v < t.h.sh_addr + t.h.sh_size; v += 4)
        {
            std::uint32_t insn = text_word(e, v);
            const char* c      = insn_class(insn);
            tally[c]++;
            std::printf("  0x%08x: %08x  %-7s", v, insn, c);
            // Prefer a real (FUNC) symbol starting here over an ISA mapping symbol.
            const std::string* best = nullptr;
            for (auto& kv : e.syms)
            {
                if (kv.second.value != v || kv.second.name.empty())
                {
                    continue;
                }
                if (kv.second.name[0] == '$' && best)
                {
                    continue;
                }
                best = &kv.second.name;
                if (kv.second.type == STT_FUNC)
                {
                    break;
                }
            }
            if (best)
            {
                std::printf("  <%s>", best->c_str());
            }
            std::printf("\n");
        }
        std::printf("\nclass tally:\n");
        for (auto& kv : tally)
        {
            std::printf("  %-8s %d\n", kv.first.c_str(), kv.second);
        }
        return 0;
    }

    // ---- Collect sites into one address-keyed map (dedup; keep the larger delay).
    std::map<std::uint32_t, Site> want;
    auto add_site = [&](std::uint32_t vaddr, std::uint32_t n, const std::string& why)
    {
        auto it = want.find(vaddr);
        if (it == want.end())
        {
            Site s;
            s.vaddr     = vaddr;
            s.n         = n;
            s.why       = why;
            want[vaddr] = s;
        }
        else if (n != it->second.n)
        {
            char b[96];
            std::snprintf(b, sizeof b, "site 0x%x requested twice (%u and %u); keeping %u", vaddr, it->second.n, n, std::max(n, it->second.n));
            warn(b);
            if (n > it->second.n)
            {
                it->second.n   = n;
                it->second.why = why;
            }
        }
    };

    for (Site& s : explicit_sites)
    {
        std::string err;
        std::uint32_t v;
        if (!resolve_loc(e, s.why, v, err))
        {
            die(err);
        }
        add_site(v, s.n, s.why);
    }

    if (!sites_file.empty())
    {
        std::ifstream f(sites_file);
        if (!f)
        {
            die("cannot open " + sites_file);
        }
        std::string line;
        int ln = 0;
        while (std::getline(f, line))
        {
            ++ln;
            size_t h = line.find('#');
            if (h != std::string::npos)
            {
                line = line.substr(0, h);
            }
            size_t b = line.find_first_not_of(" \t\r\n");
            if (b == std::string::npos)
            {
                continue;
            }
            size_t en = line.find_last_not_of(" \t\r\n");
            line      = line.substr(b, en - b + 1);
            size_t eq = line.find('=');
            if (eq == std::string::npos)
            {
                die(sites_file + ":" + std::to_string(ln) + ": want LOC=N");
            }
            std::uint32_t n;
            std::string err, v_s = line.substr(0, eq);
            if (!parse_u32(line.substr(eq + 1), n))
            {
                die(sites_file + ":" + std::to_string(ln) + ": bad N");
            }
            std::uint32_t v;
            if (!resolve_loc(e, v_s, v, err))
            {
                die(sites_file + ":" + std::to_string(ln) + ": " + err);
            }
            add_site(v, n, v_s);
        }
    }

    for (EveryReq& r : every)
    {
        int hits = 0;
        for (std::uint32_t v = t.h.sh_addr; v < t.h.sh_addr + t.h.sh_size; v += 4)
        {
            std::uint32_t insn = text_word(e, v);
            if (opcode(insn) == OP_AUIPC)
            {
                continue; // never detourable
            }
            const char* c = insn_class(insn);
            bool match = r.cls == "all" || r.cls == c || (r.cls == "call" && opcode(insn) == OP_JAL && rd(insn)) || (r.cls == "jal" && opcode(insn) == OP_JAL);
            if (match)
            {
                add_site(v, r.n, "every:" + r.cls);
                ++hits;
            }
        }
        std::fprintf(stderr, "ttnop: --every %s matched %d site(s)\n", r.cls.c_str(), hits);
    }

    if (want.empty())
    {
        die("no injection sites given (try --list, a SITE, or --every)");
    }

    // ---- Validate sites, then lay out and assemble the cave.
    std::vector<Site> sites;
    for (auto& kv : want)
    {
        sites.push_back(kv.second);
    }
    for (Site& s : sites)
    {
        validate_site(e, s);
    }

    std::uint32_t cave_vaddr = ((t.h.sh_addr + t.h.sh_size) + 3u) & ~3u;
    std::vector<Block> blocks;
    std::uint32_t cv = cave_vaddr;
    for (Site& s : sites)
    {
        Block b;
        b.site       = s.vaddr;
        b.orig       = text_word(e, s.vaddr);
        b.n          = s.n;
        b.cave_vaddr = cv;
        cv += (s.n + body_words(b.orig)) * 4;
        blocks.push_back(b);
    }
    std::uint32_t cave_size = cv - cave_vaddr;

    // Refuse if the cave would run past .text's segment into a higher region.
    for (const Elf32_Phdr& p : e.phdrs)
    {
        if (p.p_type == PT_LOAD && p.p_vaddr >= cave_vaddr && p.p_vaddr < cave_vaddr + cave_size)
        {
            char b[160];
            std::snprintf(
                b,
                sizeof b,
                "cave [0x%x,0x%x) would overlap the segment at 0x%x. reduce the delay "
                "or instrument fewer sites",
                cave_vaddr,
                cave_vaddr + cave_size,
                p.p_vaddr);
            die(b);
        }
    }

    for (Block& b : blocks)
    {
        build_block(b);
    }

    std::vector<std::uint8_t> cave_bytes;
    for (Block& b : blocks)
    {
        for (std::uint32_t wd : b.words)
        {
            size_t at = cave_bytes.size();
            cave_bytes.resize(at + 4);
            std::memcpy(cave_bytes.data() + at, &wd, 4);
        }
    }

    if (do_list)
    {
        std::printf("planned %zu detour(s), cave 0x%x..0x%x (%u bytes):\n", blocks.size(), cave_vaddr, cave_vaddr + cave_size, cave_size);
        for (Block& b : blocks)
        {
            std::printf("  site 0x%08x %-7s delay=%-6u -> cave 0x%08x\n", b.site, insn_class(b.orig), b.n, b.cave_vaddr);
        }
    }

    if (out_path.empty())
    {
        if (do_list)
        {
            return 0;
        }
        die("no -o output path given");
    }

    write_out(e, blocks, cave_vaddr, cave_bytes, out_path);

    std::uint32_t total = 0;
    for (Site& s : sites)
    {
        total += s.n;
    }
    std::fprintf(
        stderr,
        "ttnop: wrote %s\n  %zu detour(s), delay total %u, cave 0x%x..0x%x (+%u bytes)\n",
        out_path.c_str(),
        blocks.size(),
        total,
        cave_vaddr,
        cave_vaddr + cave_size,
        cave_size);

    if (do_verify)
    {
        Elf v;
        load_elf(out_path, v);
        for (Block& b : blocks)
        {
            if (text_word(v, b.site) != b.patch)
            {
                die("verify: site 0x" + std::to_string(b.site) + " patch mismatch");
            }
            const Section* cs = sec_containing(v, b.cave_vaddr);
            if (!cs)
            {
                die("verify: cave not found in any section");
            }
            for (size_t k = 0; k < b.words.size(); ++k)
            {
                size_t off = cs->h.sh_offset + (b.cave_vaddr + k * 4 - cs->h.sh_addr);
                if (rd_at<std::uint32_t>(v.buf, off) != b.words[k])
                {
                    die("verify: cave word mismatch at 0x" + std::to_string(b.cave_vaddr));
                }
            }
        }
        std::fprintf(stderr, "ttnop: verify OK (%zu detours re-checked)\n", blocks.size());
    }
    return 0;
}

int main(int argc, char** argv)
{
    return run(argc, argv);
}
